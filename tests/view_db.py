# to run
# python view_db.py ~/.mem0/persistent_test_db/history.db

import sqlite3
import sys
import os
from tabulate import tabulate  # You may need to install: pip install tabulate

def view_sqlite_database(db_path):
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        print("No tables found in the database.")
        return
    
    print(f"Tables in {db_path}:")
    for i, table in enumerate(tables):
        print(f"{i+1}. {table[0]}")
    
    table_idx = int(input("\nEnter table number to view: ")) - 1
    table_name = tables[table_idx][0]
    
    # Get table schema
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"\nSchema for {table_name}:")
    print(tabulate([[col[1], col[2]] for col in columns], headers=["Column", "Type"]))
    
    # Get data
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 100;")
    rows = cursor.fetchall()
    headers = [col[1] for col in columns]
    
    print(f"\nData in {table_name} (first 100 rows):")
    print(tabulate(rows, headers=headers, tablefmt="psql"))
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_db.py <path_to_sqlite_file>")
        sys.exit(1)
    
    view_sqlite_database(sys.argv[1])