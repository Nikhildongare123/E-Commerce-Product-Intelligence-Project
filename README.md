E-COMMERCE-PROJECT/
│
├── 📂 dataset/                              # Raw data files
│   ├── interactions.csv                     # User-product interactions
│   ├── products.csv                         # Product catalog
│   ├── purchases.csv                        # Transaction records
│   ├── reviews.csv                          # Product reviews
│   ├── sessions.csv                         # User session logs
│   └── users.csv                            # User demographics
│
├── 📂 cleaned_data/                         # Preprocessed datasets
│   ├── interactions_cleaned.csv
│   ├── products_cleaned.csv
│   ├── purchases_cleaned.csv
│   ├── reviews_cleaned.csv
│   ├── sessions_cleaned.csv
│   └── users_cleaned.csv
│
├── 📂 Notebook/                             # Data understanding & EDA
│   ├── 01_Data_Understanding_users.ipynb
│   ├── 02_Data_Understanding_Products.ipynb
│   ├── 03_EDA_Products.ipynb
│   ├── 04_Data_Understanding_Sessions.ipynb
│   ├── 05_Data_Understanding_purchases.ipynb
│   ├── 06_Data_Understanding_interactions.ipynb
│   └── 07_Data_Understanding_reviews.ipynb
│
├── 📂 Customer_Analytics/                   # Customer analytics module
│   └── Customer_analytics.ipynb             # RFM, CLV, churn analysis
│
├── 📂 Sentiment_Analysis/                   # NLP sentiment analysis
│   ├── Sentiment_Analysis.ipynb             # Model training notebook
│   ├── sentiment_model.pkl                  # Trained classifier
│   ├── tfidf.pkl                            # TF-IDF vectorizer
│   └── README.md                            # Model documentation
│
└── 📂 database_mysql/                       # Database management
    └── create_database_mysql.py             # MySQL schema & data import
