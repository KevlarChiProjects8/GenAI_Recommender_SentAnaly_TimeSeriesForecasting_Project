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

# Install dependencies
pip install -r requirements.txt

# Start Flask server
-Download the RecommenderSystem_AzureNotebook and utilize a similar review dataset (see Datasets Section down below), then run all the code cells; at the end, I used joblib to upload the count_vectorizer, cosine_similarities, and the dataset (see Notebooks section below, Azure Notebook).
-creating a "Models" folder in Vscode with a .pkl cosine_similarities (from running the Azure Notebook (RecommenderSystem_AzureNotebook) (See Notebooks Section below); the count_vectorizer isn't necessary at all
-Put your dataset in the models folder as well
-Run "python app/app.py" in Vscode terminal after completing the above

# Using Postman to test the Flask API
-After running "python app/app.py", there should be a link to the API website, and use Postman with "(url to website from Vscode terminal)/products" (Set the type of request to GET), "(url to website from Vscode terminal)/recommend" (Set the type of request to POST), and "(url to website from Vscode terminal)/analyze_sentiment" (Set the type of request to POST). 
-For each request page, in the "Headers" section, make sure to add a "Key": "Content-Type", and "Value": "application/json".
-MAKE SURE to set the data format to raw in the Body section
-For the /recommend page, put {
  "selected_ids": [a, b, c]
}        
into the Body area, substituting a, b, and c (and you can add or have less selected_ids) based off of the Vscode terminal outputting the id of the products you chose via the /products postman GET request. 
-The other two, /products and /analyze_sentiment page, don't need input, just to set the data format to raw in the Body section 9above) and the "Key": "Content-Type", "Value": "application/json" in the Headers section.

### Notebooks
- **Azure Notebook** – Implements Generative AI (Gemini Flash 2.0), Sentiment Analysis, and Recommender System with Flask API. Ignore the Azure Blob Cloud Storage commands and replace it with df = pd.read_csv("(Your file name)", low_memory=False). Try to aim for a dataset no larger than 100,000 reviews to avoid crashing. Run in Google Colab.
- **Colab Notebook** – Focuses on Exploratory Data Analysis (EDA) and Time-Series Forecasting with XGBoost and Prophet. Run in Google Colab

### Datasets
- df_meta was too large to upload, and the df_review dataset was lost, but any similar Amazon review dataset with product titles, reviews, and date columns will work with the code as long as the respective variable names are changed, e.g. for the RecommenderSystem_AzureNotebook, if it were a similar dataset with an adjacent title column name 'product_name', change the column 'product_name' to title using the pd.rename({'Old Column'}: {'title'}, inplace=True) command.
