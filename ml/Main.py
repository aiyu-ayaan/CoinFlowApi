import json
from dataclasses import dataclass, is_dataclass, asdict
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from datetime import datetime

from orjson import orjson
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


########## Function to load and preprocess data ###################################################
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))

    customers = pd.read_csv(os.path.join(base_dir, 'customers_data.csv'))
    orders = pd.read_csv(os.path.join(base_dir, 'orders_data.csv'))
    order_items = pd.read_csv(os.path.join(base_dir, 'order_items_data.csv'))
    products = pd.read_csv(os.path.join(base_dir, 'products_data.csv'))
    orders['order_date'] = pd.to_datetime(orders['order_date'], format='%d-%m-%Y')

    return customers, orders, order_items, products


####### Function to merge datasets and create features ###################################################
def create_features(customers, orders, order_items, products):
    """Merge datasets and create features for modeling"""
    # Merge order_items with products to get product details
    order_products = pd.merge(order_items, products, on='product_id')
    # Merge orders with customers
    customer_orders = pd.merge(orders, customers, on='customer_id')
    # Create full transaction dataset
    transactions = pd.merge(customer_orders, order_products, on='order_id')
    # Calculate time between orders for each customer
    customer_order_dates = orders.sort_values(['customer_id', 'order_date'])
    customer_order_dates['prev_order_date'] = customer_order_dates.groupby('customer_id')['order_date'].shift(1)
    customer_order_dates['days_since_prev_order'] = (customer_order_dates['order_date'] -
                                                     customer_order_dates['prev_order_date']).dt.days

    return transactions, customer_order_dates


####### Extract time-based features ######################################################
def extract_time_features(df):
    """Extract time-based features from datetime column"""
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_weekday'] = df['order_date'].dt.weekday
    df['order_quarter'] = df['order_date'].dt.quarter
    return df


# 1. Model to predict customer's next purchase item ###################################################
def build_next_item_model(transactions):
    """Build a model to predict a customer's next purchase item"""
    # Create customer-product purchase history matrix
    purchase_matrix = transactions.pivot_table(
        index='customer_id',
        columns='product_id',
        values='quantity',
        aggfunc='sum',
        fill_value=0
    )

    # For each customer, keep track of their purchase sequence
    customer_product_sequence = transactions.sort_values(['customer_id', 'order_date'])
    customer_product_sequence['prev_product_id'] = customer_product_sequence.groupby('customer_id')['product_id'].shift(
        1)

    # Remove rows with NaN prev_product_id (first purchases)
    product_transition = customer_product_sequence.dropna(subset=['prev_product_id'])

    # Create product transition matrix
    product_transition_matrix = pd.crosstab(
        product_transition['prev_product_id'].astype(int),
        product_transition['product_id']
    )

    # Function to predict next item based on previous item
    def predict_next_item(prev_product_id):
        if prev_product_id in product_transition_matrix.index:
            # Get probabilities of next products based on previous product
            next_product_probs = product_transition_matrix.loc[prev_product_id]
            # Return product with highest probability
            if next_product_probs.sum() > 0:
                return next_product_probs.idxmax()
        # If no data for this product, return most popular product overall
        return transactions['product_id'].value_counts().idxmax()

    # Create a recommendation system using collaborative filtering
    # Initialize the KNN model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(purchase_matrix)

    def recommend_items_for_customer(customer_id, n_recommendations=1):
        """Recommend items for a specific customer based on similar customers"""
        if customer_id not in purchase_matrix.index:
            # If customer is not in the matrix, return most popular products
            popular_products = transactions['product_id'].value_counts().head(n_recommendations).index.tolist()
            return popular_products

        # Get the customer's purchase vector
        customer_vector = purchase_matrix.loc[customer_id].values.reshape(1, -1)

        # Find similar customers
        distances, indices = model_knn.kneighbors(customer_vector, n_neighbors=5)

        # Get similar customer IDs
        similar_customer_indices = indices.flatten()[1:]  # Exclude the customer itself
        similar_customers = [purchase_matrix.index[i] for i in similar_customer_indices]

        # Find products that similar customers bought but the current customer hasn't
        recommended_items = []
        for sim_cust in similar_customers:
            sim_cust_products = purchase_matrix.loc[sim_cust]
            # Products the similar customer bought
            bought_products = sim_cust_products[sim_cust_products > 0].index.tolist()
            # Products the target customer hasn't bought
            target_customer_products = purchase_matrix.loc[customer_id]
            not_bought = [p for p in bought_products if target_customer_products[p] == 0]
            recommended_items.extend(not_bought)

        # Count occurrences and get top recommendations
        if recommended_items:
            product_counts = pd.Series(recommended_items).value_counts()
            top_recommendations = product_counts.head(n_recommendations).index.tolist()
            return top_recommendations
        else:
            # Fallback to popular products
            popular_products = transactions.loc[
                transactions['customer_id'] != customer_id, 'product_id'
            ].value_counts().head(n_recommendations).index.tolist()
            return popular_products

    # Return the prediction functions
    return predict_next_item, recommend_items_for_customer, product_transition_matrix


# 2. Model to predict customer's next purchase date ###################################################
def build_next_purchase_date_model(customer_order_dates, customers):
    """Build a model to predict when a customer will make their next purchase"""
    # Remove rows with NaN days_since_prev_order (first purchases)
    purchase_intervals = customer_order_dates.dropna(subset=['days_since_prev_order'])

    # Merge with customer data for more features
    purchase_intervals = pd.merge(
        purchase_intervals,
        customers[['customer_id', 'total_orders', 'avg_order_value']],
        on='customer_id'
    )

    # Extract time features
    purchase_intervals = extract_time_features(purchase_intervals)

    # Define features and target
    features = [
        'customer_id', 'order_total', 'total_orders', 'avg_order_value',
        'order_month', 'order_day', 'order_weekday', 'order_quarter'
    ]

    # Add payment method if it's categorical
    if 'payment_method' in purchase_intervals.columns:
        # One-hot encode payment method
        purchase_intervals = pd.get_dummies(purchase_intervals, columns=['payment_method'], drop_first=True)
        payment_features = [col for col in purchase_intervals.columns if 'payment_method' in col]
        features.extend(payment_features)

    X = purchase_intervals[features]
    y = purchase_intervals['days_since_prev_order']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create preprocessor
    # We need to use indices, not column names for the ColumnTransformer
    # Create a list of indices for numerical features
    numerical_indices = [i for i, f in enumerate(features) if f != 'customer_id']
    categorical_indices = []

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_indices),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices)
        ]
    )

    # Create and train the model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Function to predict days until next purchase for a customer
    def predict_next_purchase_date(customer_info):
        """Predict days until next purchase for a given customer"""
        # Convert customer_info to numpy array to ensure proper shape
        customer_info_array = np.array(customer_info).reshape(1, -1)
        prediction = model.predict(customer_info_array)[0]
        return max(1, round(prediction))  # Ensure prediction is at least 1 day

    # Calculate average purchase interval for each customer (as a fallback)
    customer_avg_intervals = purchase_intervals.groupby('customer_id')['days_since_prev_order'].mean()

    def get_next_purchase_date(customer_id, last_purchase_date=None):
        """Get predicted next purchase date for a customer"""
        # Get customer's last order information
        if last_purchase_date is None:
            customer_orders = customer_order_dates[customer_order_dates['customer_id'] == customer_id]
            if customer_orders.empty:
                # If no order history, use overall average
                days_to_add = purchase_intervals['days_since_prev_order'].mean()
                return (datetime.now() + pd.Timedelta(days=days_to_add))

            last_purchase_date = customer_orders['order_date'].max()

        # Get customer info for prediction
        customer_data = customers[customers['customer_id'] == customer_id]
        if customer_data.empty:
            # If customer not found, use average interval
            if customer_id in customer_avg_intervals:
                days_to_add = customer_avg_intervals[customer_id]
            else:
                days_to_add = purchase_intervals['days_since_prev_order'].mean()
            return last_purchase_date + pd.Timedelta(days=days_to_add)

        # Get last order information
        last_order = customer_order_dates[
            (customer_order_dates['customer_id'] == customer_id) &
            (customer_order_dates['order_date'] == last_purchase_date)
            ]

        if last_order.empty:
            # Use customer average if specific order not found
            if customer_id in customer_avg_intervals:
                days_to_add = customer_avg_intervals[customer_id]
            else:
                days_to_add = purchase_intervals['days_since_prev_order'].mean()
        else:
            # Prepare features for prediction
            last_order_info = last_order.iloc[0]

            # Create feature vector
            customer_info = [
                customer_id,
                last_order_info.get('order_total', customer_data['avg_order_value'].iloc[0]),
                customer_data['total_orders'].iloc[0],
                customer_data['avg_order_value'].iloc[0],
                last_purchase_date.month,
                last_purchase_date.day,
                last_purchase_date.weekday(),
                (last_purchase_date.month - 1) // 3 + 1  # Quarter
            ]

            # Since we're using numerical indices now, we'll handle payment method differently
            # Just add the payment method value if it exists in the last order
            if 'payment_method' in last_order_info:
                payment_method = last_order_info['payment_method']
                # We'll need to convert this to a numeric value based on our encoding
                # For simplicity, we'll just use a default value of 1
                customer_info.append(1)

            # Predict days until next purchase
            days_to_add = predict_next_purchase_date(customer_info)

        # Return predicted next purchase date
        predicted_date = last_purchase_date + pd.Timedelta(days=days_to_add)
        # If predicted date is in the past, shift it to today + days_to_add
        if predicted_date < pd.Timestamp.today().normalize():
            predicted_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=days_to_add)
        # Return the adjusted date
        return predicted_date

    return model, get_next_purchase_date


# 3. Model to predict overall most purchased item ###################################################
def predict_most_purchased_item(transactions):
    """Predict the most purchased item overall and by category"""
    # Calculate total quantity purchased for each product
    product_quantities = transactions.groupby(['product_id', 'product_name'])['quantity'].sum().reset_index()
    product_quantities = product_quantities.sort_values('quantity', ascending=False)

    # Overall most purchased item
    overall_most_purchased = product_quantities.iloc[0]

    # Most purchased items by category
    category_most_purchased = transactions.groupby(['category', 'product_id', 'product_name'])[
        'quantity'].sum().reset_index()
    category_most_purchased = category_most_purchased.sort_values(['category', 'quantity'], ascending=[True, False])
    top_by_category = category_most_purchased.groupby('category').first()

    # Identify trends over time
    transactions['month_year'] = transactions['order_date'].dt.to_period('M')
    monthly_trends = transactions.groupby(['month_year', 'product_id', 'product_name'])['quantity'].sum().reset_index()
    monthly_top_products = monthly_trends.sort_values(['month_year', 'quantity'], ascending=[True, False])
    monthly_top_product = monthly_top_products.groupby('month_year').first()

    return overall_most_purchased, top_by_category, monthly_top_product


@dataclass
class ShoppingPredictions:
    """
    A class to store and manage all customer purchase predictions.
    """
    next_item: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None
    next_purchase_date: Optional[Dict[str, Any]] = None
    # product_transition_matrix: Optional[pd.DataFrame] = None
    # overall_most_purchased: Optional[pd.Series] = None
    # top_by_category: Optional[pd.DataFrame] = None
    # monthly_top_product: Optional[pd.DataFrame] = None


def predict_next_item_for_customer(predict_next_item, transactions, products, customer_id):
    """Predict the next item a customer will purchase."""
    # sample_customer_last_product = \
    #     transactions[transactions['customer_id'] == customer_id].sort_values('order_date').iloc[-1]['product_id']
    # predicted_next_item = predict_next_item(sample_customer_last_product)
    # product_name = products[products['product_id'] == predicted_next_item]['product_name'].iloc[0]
    # return {
    #     "product_id": predicted_next_item,
    #     "product_name": product_name
    # }
    filtered = transactions[transactions['customer_id'] == customer_id].sort_values('order_date')

    if not filtered.empty:
        sample_customer_last_product = filtered.iloc[-1]['product_id']
        predicted_next_item = predict_next_item(sample_customer_last_product)
        product_row = products[products['product_id'] == predicted_next_item]
        if not product_row.empty:
            product_name = product_row['product_name'].iloc[0]
            # print(f"Customer {sample_customer_id} next purchase item prediction: {product_name} (ID: {predicted_next_item})")
            return {
            "product_id": predicted_next_item,
            "product_name": product_name
            }
        else:
            return f"Predicted product ID {predicted_next_item} not found in product catalog."
    else:
        return f"No transactions found for customer ID: {customer_id} in the record"
        


def get_recommendations_for_customer(recommend_items, products, customer_id, num_recommendations=3):
    """Get recommended items for a customer."""
    recommended_items = recommend_items(customer_id, num_recommendations)
    recommended_names = products[products['product_id'].isin(recommended_items)]['product_name'].tolist()
    return {
        "product_ids": recommended_items.tolist() if hasattr(recommended_items, 'tolist') else recommended_items,
        "product_names": recommended_names
    }


def predict_next_purchase_date_for_customer(get_next_purchase_date, orders, customer_id):
    """Predict when a customer will make their next purchase."""
    sample_customer_last_date = orders[orders['customer_id'] == customer_id]['order_date'].max()
    next_purchase_date = get_next_purchase_date(customer_id, sample_customer_last_date)
    if pd.notna(next_purchase_date):
        return {
        "date": next_purchase_date.strftime('%d-%m-%Y'),
        "date_obj": next_purchase_date
    }
    else:
        return f"{customer_id} next purchase date is not available."
    


def predict_data(customer_id :int) -> str:
    # Load data
    customers, orders, order_items, products = load_data()
    # Create features
    transactions, customer_order_dates = create_features(customers, orders, order_items, products)
    # 1. Build model to predict next purchase item
    predict_next_item, recommend_items, product_transition_matrix = build_next_item_model(transactions)
    # 2. Build model to predict next purchase date
    next_date_model, get_next_purchase_date = build_next_purchase_date_model(customer_order_dates, customers)
    # 3. Predict most purchased item
    overall_most_purchased, top_by_category, monthly_top_product = predict_most_purchased_item(transactions)
    # Get a sample customer

    # Use the functions to get predictions
    next_item = predict_next_item_for_customer(predict_next_item, transactions, products, customer_id)
    recommendations = get_recommendations_for_customer(recommend_items, products, customer_id)
    next_date = predict_next_purchase_date_for_customer(get_next_purchase_date, orders, customer_id)
    # Return results wrapped in the ShoppingPredictions class
    prediction = ShoppingPredictions(
        next_item=next_item,
        recommendations=recommendations,
        next_purchase_date=next_date
        # product_transition_matrix=product_transition_matrix,
        # overall_most_purchased=overall_most_purchased,
        # top_by_category=top_by_category,
        # monthly_top_product=monthly_top_product
    )
    results_dict = stringify_keys(prediction.__dict__)
    json_str = json.dumps(results_dict, cls=EnhancedJSONEncoder)
    return json_str


def default_serializer(obj):
    """Custom JSON serializer for numpy and pandas objects."""
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        if obj.name is None:
            return obj.to_dict()
        return {str(obj.name): obj.to_dict()}  # Ensure key is string
    elif isinstance(obj, pd.DataFrame):
        # Convert both index and columns to strings
        return {
            'index': obj.index.astype(str).tolist(),
            'columns': obj.columns.astype(str).tolist(),
            'data': obj.values.tolist()
        }
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        # Convert all keys to strings
        return {str(k): default_serializer(v) for k, v in obj.items()}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        return default_serializer(obj)


def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [stringify_keys(x) for x in obj]
    return obj


if __name__ == "__main__":
    print(predict_data(1060685))

