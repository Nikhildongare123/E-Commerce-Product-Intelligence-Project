# 🛒 E-Commerce Customer Analytics Platform
An end-to-end data analytics and machine learning solution for e-commerce customer insights, featuring comprehensive data analysis, sentiment analysis, and predictive modeling.
# 📁 Project Structure
E-COMMERCE-PROJECT/
│
├── dataset/                          # Raw data files
│   ├── interactions.csv
│   ├── products.csv
│   ├── purchases.csv
│   ├── reviews.csv
│   ├── sessions.csv
│   └── users.csv
│
├── cleaned_data/                     # Preprocessed datasets
│   ├── interactions_cleaned.csv
│   ├── products_cleaned.csv
│   ├── purchases_cleaned.csv
│   ├── reviews_cleaned.csv
│   ├── sessions_cleaned.csv
│   └── users_cleaned.csv
│
├── Notebook/                         # Data understanding & EDA notebooks
│   ├── 01_Data_Understanding_users.ipynb
│   ├── 02_Data_Understanding_Products.ipynb
│   ├── 03_EDA_Products.ipynb
│   ├── 04_Data_Understanding_Sessions.ipynb
│   ├── 05_Data_Understanding_purchases.ipynb
│   ├── 06_Data_Understanding_interactions.ipynb
│   └── 07_Data_Understanding_reviews.ipynb
│
├── Customer_Analytics/               # Customer analytics module
│   └── Customer_analytics.ipynb
│
├── Sentiment_Analysis/               # Sentiment analysis module
│   ├── Sentiment_Analysis.ipynb
│   ├── sentiment_model.pkl
│   ├── tfidf.pkl
│   └── README.md
│
└── database_mysql/                   # Database management
    └── create_database_mysql.py
# 🚀 Features
1. Data Pipeline
Raw data ingestion from CSV files
Comprehensive data cleaning and preprocessing
Automated data transformation pipeline

2. Exploratory Data Analysis (EDA)
3. User Analysis: Demographic patterns, behavior analysis
Product Analysis: Category performance, pricing insights
Session Analysis: User engagement, bounce rates
Purchase Analysis: Revenue trends, seasonality
Interaction Analysis: Click patterns, conversion funnels
Review Analysis: Rating distributions, feedback patterns

3. Customer Analytics
Customer segmentation (RFM analysis)
Customer Lifetime Value (CLV) prediction
Churn prediction and retention strategies
Behavioral cohort analysis

4. Sentiment Analysis
Product review sentiment scoring using NLP
Custom-trained sentiment classification model
TF-IDF vectorization for feature extraction
Real-time sentiment prediction on new reviews
