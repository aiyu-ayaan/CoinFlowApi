import json
from dataclasses import is_dataclass, asdict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from orjson import orjson

from ml.Main import predict_next_item_for_customer, get_recommendations_for_customer, \
    predict_next_purchase_date_for_customer, ShoppingPredictions, create_features, \
    build_next_purchase_date_model, build_next_item_model, predict_most_purchased_item, load_data, stringify_keys, \
    EnhancedJSONEncoder, default_serializer, predict_data

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello Nirvikar'


customers, orders, order_items, products = load_data()
transactions, customer_order_dates = create_features(customers, orders, order_items, products)
predict_next_item, recommend_items, product_transition_matrix = build_next_item_model(transactions)
next_date_model, get_next_purchase_date = build_next_purchase_date_model(customer_order_dates, customers)
overall_most_purchased, top_by_category, monthly_top_product = predict_most_purchased_item(transactions)


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get data from the POST request
#     data = request.get_json()
#
#     # Check if customer_id exists in request
#     if 'customer_id' not in data:
#         return jsonify({"error": "Missing customer_id in request"}), 400
#
#     customer_id = data['customer_id']
#
#     # Check if customer exists
#     if customer_id not in customers['customer_id'].values:
#         return jsonify({"error": f"Customer ID {customer_id} not found"}), 404
#
#
#     try:
#         # Generate predictions
#         next_item = predict_next_item_for_customer(predict_next_item, transactions, products, customer_id)
#
#         recommendations = get_recommendations_for_customer(recommend_items, products, customer_id,
#                                                            num_recommendations=data.get('num_recommendations', 3))
#
#         next_date = predict_next_purchase_date_for_customer(get_next_purchase_date, orders, customer_id)
#
#
#         # Create shopping predictions
#         # print(f'{type(next_item)} {type(recommendations)} {type(next_date)} {type(overall_most_purchased)} {type(top_by_category)} {type(monthly_top_product)}')
#
#         predictions = ShoppingPredictions(
#             next_item=next_item,
#             recommendations=recommendations,
#             next_purchase_date=next_date,
#             models_status="Predictions generated successfully.",
#             overall_most_purchased=overall_most_purchased,
#             top_by_category=top_by_category,
#             monthly_top_product=monthly_top_product
#         )
#
#         results_dict = stringify_keys(predictions.__dict__)
#         json_str = json.dumps(results_dict, cls=EnhancedJSONEncoder)
#         print(json_str)
#         return jsonify(json_str)
#
#     except Exception as e:
#         print(f'Error: {str(e)}')
#         return jsonify({"error": f"Error generating predictions: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'customer_id' not in data:
            return jsonify({"error": "Missing customer_id in request"}), 400

        customer_id = data['customer_id']
        if customer_id not in customers['customer_id'].values:
            return jsonify({"error": f"Customer ID {customer_id} not found"}), 404
        # Check if customer_id is a valid integer
        print(customer_id)
        # Generate predictions
        predicted_data = predict_data(customer_id)

        return predicted_data

    except Exception as e:
        app.logger.error(f'Error: {str(e)}', exc_info=True)
        return jsonify({"error": f"Error generating predictions: {str(e)}"}), 500


@app.route('/top_products', methods=['POST'])
def get_top_products():
    try:
        # Return overall most purchased products and by category
        result = {
            "overall_most_purchased": {
                "product_id": int(overall_most_purchased['product_id']),
                "product_name": overall_most_purchased['product_name'],
                "quantity": int(overall_most_purchased['quantity'])
            },
            "top_by_category": top_by_category.reset_index().to_dict('records')
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error retrieving top products: {str(e)}"}), 500


@app.route('/monthly_trends', methods=['POST'])
def get_monthly_trends():
    try:
        # Convert Period objects to strings for JSON serialization
        trends = []
        for month, row in monthly_top_product.reset_index().iterrows():
            trends.append({
                "month": str(row['month_year']),
                "product_id": int(row['product_id']),
                "product_name": row['product_name'],
                "quantity": int(row['quantity'])
            })
        return jsonify({"monthly_trends": trends})
    except Exception as e:
        return jsonify({"error": f"Error retrieving monthly trends: {str(e)}"}), 500


if __name__ == '__main__':
    app.run()
