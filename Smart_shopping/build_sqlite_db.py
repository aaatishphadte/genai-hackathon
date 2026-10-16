# build_sqlite_db.py
#
# Loads both CSVs into db/ecommerce.db and creates the long-term memory tables.

import os
import sqlite3
import pandas as pd
import ast

import config
from utils.db import init_catalog_tables, init_memory_tables

# Load data
customer_df = pd.read_csv(config.CUSTOMER_CSV)
product_df = pd.read_csv(config.PRODUCT_CSV)

# Clean and parse list columns
customer_df['Browsing_History'] = customer_df['Browsing_History'].apply(ast.literal_eval)
customer_df['Purchase_History'] = customer_df['Purchase_History'].apply(ast.literal_eval)
product_df['Similar_Product_List'] = product_df['Similar_Product_List'].apply(ast.literal_eval)

# Connect to SQLite
os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
conn = sqlite3.connect(config.DB_PATH)
cur = conn.cursor()

# Create tables
init_catalog_tables(conn)

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
init_memory_tables(conn)
conn.close()

print(f"SQLite database created successfully at {config.DB_PATH}")
