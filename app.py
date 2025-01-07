from flask import Flask, request, jsonify
import pandas as pd
import joblib 
from sklearn.metrics.pairwise import cosine_similarity 
import google.generativeai as genai
from google.api_core import retry
import os
from flask_cors import CORS 
import numpy as np
import re 
import random

# Initalize Flask App
app = Flask(__name__)
CORS(app)

# Load models and data
vectorizer = joblib.load("models/count_vectorizer.pkl")
similarity_matrix = joblib.load("models/cosine_similarity_scores.pkl")
df_meta = pd.read_csv("models/df_meta.csv", low_memory=False)

# Configure Google Generative AI
os.environ["GOOGLE_API_KEY"] = "AIzaSyBwercL0gOt3BF8AEJRDyYpeELG6sMBPhs"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

retry_policy = {
    "retry": retry.Retry(predicate=retry.if_transient_error, initial=10, multiplier=1.5, timeout=300)
}
recommendations = None  # Initialize recommendations

# clean Text Data
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = text.strip()  # Trim spaces
    return text

df_meta['title'] = df_meta['title'].astype(str).apply(clean_text)
df_meta['main_category'] = df_meta['main_category'].astype(str).apply(clean_text)
df_meta['details'] = df_meta['details'].astype(str).apply(clean_text)

# Generate Recommendations Function
def generate_recommendations(picked_product_ids, meta_df, all_similarity, n):
    valid_indices = []
    for product in picked_product_ids: 
        if product <= meta_df.shape[0]:
            valid_indices.append(product)

        if len(valid_indices) == 0:
            return []
        
        query_vectors = np.array([all_similarity[index] for index in valid_indices])

        if len(query_vectors) == 0:
            return []
        
        if len(query_vectors) > 1:
            query_vectors = np.mean(query_vectors, axis=0, keepdims=True)

        similarities = cosine_similarity(query_vectors, all_similarity)
        similar_indices = np.argsort(similarities[0])[::-1][1:n+1]
        similar_products = meta_df['product_id'].iloc[similar_indices].values.tolist()

        return similar_products 
    

# Route 1: Get 10 Random Products
@app.route('/products', methods=['GET'])
def get_random_products():
    random_products = df_meta.sample(10).to_dict(orient='records')
    return jsonify(random_products)

# Route 2: Recommend Top 5 Similar Products
# Create a 'product_id' column

# Create a 'product_id' column
df_meta = df_meta.reset_index()
df_meta.rename(columns={'index': 'product_id'}, inplace=True) #Rename it from index to product_id\

@app.route('/recommend', methods=['POST'])
def recommend_products():
    global recommendations
    try:
        data = request.json
        print("Request Data:", data)  # Debugging log

        selected_ids = data.get('selected_ids', [])
        print("Selected IDs:", selected_ids)  # Debugging log

        # Check if IDs are valid
        print("Dataframe Columns:", df_meta.columns)  # Debugging log
        print("Sample Data:", df_meta.head())  # Debugging log

        # Ensure the 'id' column exists
        if 'product_id' not in df_meta.columns:
            return jsonify({"error": "ID column missing in metadata"}), 500

        # Create ID-to-index mapping
        product_id_to_index = {product_id: i for i, product_id in enumerate(df_meta['product_id'])}

        # Filter valid IDs
        valid_ids = [pid for pid in selected_ids if pid in product_id_to_index]
        print("Valid IDs:", valid_ids)  # Debugging log

        if not valid_ids:
            return jsonify({"error": "No valid product IDs found"}), 400

        # Generate recommendations
        recommendations = generate_recommendations(valid_ids, df_meta, similarity_matrix, n=5)
        recommended_products = df_meta[df_meta['product_id'].isin(recommendations)].to_dict(orient='records') # In pandas, the isin() method is used to check whether each element in a DataFrame or Series is contained in a list of values. It returns a boolean Series or DataFrame indicating whether each element is present in the provided values.

        print("Recommended Products:", recommended_products)  # Debugging log

        return jsonify(recommended_products)

    except Exception as e:
        print("Error:", str(e))  # Log error details
        return jsonify({"error": str(e)}), 500


# Route 3: aspect-Based Sentiment Analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    global recommendations

    # Check if recommendations exist
    if recommendations is None or len(recommendations) == 0:
        return jsonify({"error": "No recommendations available. Please call /recommend first."}), 400

    # Create a dictionary with product name and blank reviews
    aspect_sentiment = {}

    # Configure the model
    model = genai.GenerativeModel(
        'gemini-2.0-flash-thinking-exp-1219',
        generation_config=genai.GenerationConfig(
            temperature=1,
            top_k=16,
            top_p=1
        )
    )

    # Generate sentiment analysis
    for i in recommendations:
        try:
            zero_shot_prompt = f"""
            Provide a concise aspect-based sentiment analysis for the following product details: 
            Title: {df_meta.iloc[i]['title']}
            Average Rating: {df_meta.iloc[i]['average_rating']}
            Details: {df_meta.iloc[i]['details']}

            Output Format:
            - Positive Aspects: [list key aspects with positive sentiment]
            - Negative Aspects: [list key aspects with negative sentiment]
            - Overall Sentiment: [positive/neutral/negative]
            """
            aspect_sentiment[df_meta.iloc[i]['title']] = model.generate_content(
                zero_shot_prompt, request_options=retry_policy
            ).text
        except Exception as e:
            aspect_sentiment[df_meta.iloc[i]['title']] = f"Error: {str(e)}"

    return jsonify(aspect_sentiment)

@app.route('/favicon.ico')
def favicon():
    return '', 204

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)