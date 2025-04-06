
import sqlite3
import pandas as pd
import ast

# Load data
customer_df = pd.read_csv("customer_data_collection.csv")
product_df = pd.read_csv("product_recommendation_data.csv")

# Clean and parse list columns
customer_df['Browsing_History'] = customer_df['Browsing_History'].apply(ast.literal_eval)
customer_df['Purchase_History'] = customer_df['Purchase_History'].apply(ast.literal_eval)
product_df['Similar_Product_List'] = product_df['Similar_Product_List'].apply(ast.literal_eval)

# Connect to SQLite
conn = sqlite3.connect("ecommerce.db")
cur = conn.cursor()

# Create tables
cur.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    location TEXT,
    browsing_history TEXT,
    purchase_history TEXT,
    customer_segment TEXT,
    avg_order_value REAL,
    holiday TEXT,
    season TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT,
    price REAL,
    brand TEXT,
    avg_rating_similar REAL,
    product_rating REAL,
    sentiment_score REAL,
    holiday TEXT,
    season TEXT,
    geo_location TEXT,
    similar_products TEXT,
    probability_of_recommendation REAL
)
""")

# Insert data into customers table
for _, row in customer_df.iterrows():
    cur.execute("""
    INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['Customer_ID'],
        row['Age'],
        row['Gender'],
        row['Location'],
        str(row['Browsing_History']),
        str(row['Purchase_History']),
        row['Customer_Segment'],
        row['Avg_Order_Value'],
        row['Holiday'],
        row['Season']
    ))

# Insert data into products table
for _, row in product_df.iterrows():
    cur.execute("""
    INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['Product_ID'],
        row['Category'],
        row['Subcategory'],
        row['Price'],
        row['Brand'],
        row['Average_Rating_of_Similar_Products'],
        row['Product_Rating'],
        row['Customer_Review_Sentiment_Score'],
        row['Holiday'],
        row['Season'],
        row['Geographical_Location'],
        str(row['Similar_Product_List']),
        row['Probability_of_Recommendation']
    ))

conn.commit()
conn.close()

print("SQLite database created successfully as ecommerce.db")
