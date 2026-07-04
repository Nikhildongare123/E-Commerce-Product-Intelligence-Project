import pymysql
import pandas as pd

# ===========================
# 1. MySQL Connection
# ===========================

connection = pymysql.connect(
    host="localhost",
    user="root",
    password="nikhil0909",   # Change if required
    autocommit=True
)

cursor = connection.cursor()

# ===========================
# 2. Create Database
# ===========================

cursor.execute("CREATE DATABASE IF NOT EXISTS EcommerceDB")
cursor.execute("USE EcommerceDB")

# ===========================
# 3. Load CSV Files
# ===========================

users = pd.read_csv("D:\\E-Commerce-Project\\cleaned_data\\users_cleaned.csv")
products = pd.read_csv("D:\\E-Commerce-Project\\cleaned_data\\products_cleaned.csv")
sessions = pd.read_csv("D:\\E-Commerce-Project\\cleaned_data\\sessions_cleaned.csv")
interactions = pd.read_csv("D:\\E-Commerce-Project\\cleaned_data\\interactions_cleaned.csv")
purchases = pd.read_csv("D:\\E-Commerce-Project\\cleaned_data\\purchases_cleaned.csv")
reviews = pd.read_csv("D:\\E-Commerce-Project\\cleaned_data\\reviews_cleaned.csv")

# ===========================
# 4. Function to Create Table
# ===========================

def create_table_from_df(df, table_name):

    cols = []

    for col in df.columns:

        dtype = str(df[col].dtype).lower()

        # Integer
        if "int" in dtype:
            sql_type = "INT"

        # Float
        elif "float" in dtype:
            sql_type = "DOUBLE"

        # Boolean
        elif "bool" in dtype:
            sql_type = "BOOLEAN"

        # Date / Time columns
        elif "date" in col.lower() or "time" in col.lower():
            sql_type = "DATETIME"

        # Long text columns
        elif (
            "description" in col.lower()
            or "review_text" in col.lower()
            or "title" in col.lower()
        ):
            sql_type = "TEXT"

        # UUID / IDs
        elif col.lower().endswith("_id"):
            sql_type = "VARCHAR(50)"

        # Remaining text columns
        else:
            max_len = df[col].astype(str).str.len().max()

            if pd.isna(max_len):
                max_len = 100

            max_len = int(max_len)

            if max_len <= 50:
                sql_type = "VARCHAR(50)"
            elif max_len <= 100:
                sql_type = "VARCHAR(100)"
            elif max_len <= 255:
                sql_type = "VARCHAR(255)"
            else:
                sql_type = "TEXT"

        cols.append(f"`{col}` {sql_type}")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        {",".join(cols)}
    )
    """

    cursor.execute(create_sql)

# ===========================
# 5. Create Tables
# ===========================

create_table_from_df(users, "users")
create_table_from_df(products, "products")
create_table_from_df(sessions, "sessions")
create_table_from_df(interactions, "interactions")
create_table_from_df(purchases, "purchases")
create_table_from_df(reviews, "reviews")

# ===========================
# 6. Function to Insert Data
# ===========================

def insert_data(df, table_name):

    df = df.where(pd.notnull(df), None)

    columns = ",".join([f"`{c}`" for c in df.columns])

    placeholders = ",".join(["%s"] * len(df.columns))

    sql = f"""
    INSERT INTO `{table_name}`
    ({columns})
    VALUES ({placeholders})
    """

    data = [tuple(row) for row in df.itertuples(index=False, name=None)]

    cursor.executemany(sql, data)

# ===========================
# 7. Insert Data
# ===========================

insert_data(users, "users")
insert_data(products, "products")
insert_data(sessions, "sessions")
insert_data(interactions, "interactions")
insert_data(purchases, "purchases")
insert_data(reviews, "reviews")

print("✅ EcommerceDB created successfully.")
print("✅ All 6 CSV files imported into MySQL.")

# ===========================
# 8. Close Connection
# ===========================

cursor.close()
connection.close()