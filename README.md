# GenAI_Recommender_SentAnaly_TimeSeriesForecasting_Project
Using Amazon meta reviews, I built a recommender system to first give users a random sample of products to choose from, and recommended products based off the products they liked.

# AI-Powered Recommender System with Sentiment Analysis and Time-Series Forecasting

### Project Overview
This project is a Flask API-based AI system for product recommendations and sentiment analysis, integrating:
- **Google Generative AI (Gemini Flash 2.0)** for **Aspect-Based Sentiment Analysis**.  
- **Time-Series Forecasting** using **Prophet** and **XGBoost** to predict Amazon review trends with **99.9% accuracy**.  
- **Explainable AI (SHAP)** to interpret models and highlight feature importance.  
- **HuggingFace Transformers** for advanced sentiment analysis.  
- **EDA and Visualization** using **Pandas, Matplotlib, and Seaborn**.  

### Features
1. **Product Recommendations** – Recommender system using **cosine similarity**.  
2. **Sentiment Analysis** – Analyze reviews using **Generative AI** and **HuggingFace**.  
3. **Time-Series Forecasting** – Predict future star ratings with engineered features.  
4. **Explainable AI** – Use **SHAP values** for interpretability.  
5. **Flask API Integration** – Built RESTful endpoints for deployment.

### Technologies
- **Python, Flask, Google Generative AI, HuggingFace, SHAP, Pandas, Scikit-learn, Prophet, XGBoost**  
- **Postman for API testing**  

### Installation
```bash
# Clone the repository
git clone https://github.com/username/AI_Recommender_Project.git
cd AI_Recommender_Project

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app/app.py

### Notebooks
- **Azure Notebook** – Implements Generative AI (Gemini Flash 2.0), Sentiment Analysis, and Recommender System with Flask API.  
- **Colab Notebook** – Focuses on Exploratory Data Analysis (EDA) and Time-Series Forecasting with XGBoost and Prophet.

### Datasets
- **Azure Notebook Datasets** - df_meta.csv supports the recommender system, and the cosine_similarity_scores.pkl and count_vectorizer.pkl are used to build the recommender system; the count_vectorizer turns text into a vector based on the frequency (count) of each word that occurs in the entire text.
- **Colab Notebook Datasets** - The df_review.csv dataset was lost, but any review dataset with date and review columns will work. To test the EDA graphs and Time-Series Forecasting, change the column names accordingly: e.g. for a similar dataset, change the 'review_date' column to 'timestamp', the 'text' column to 'review_body', etc.
                              - The df_meta.csv dataset for the Azure Notebook Dataset will work for the Colab Notebook EDA graphs.
