import sqlite3
from datetime import date, timedelta

# Sets up database for history of user queries and the results
def main():
    connection = sqlite3.connect('user.db')
    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS queries;")
    cursor.execute("DROP TABLE IF EXISTS results;")

    cursor.execute("""
    CREATE TABLE queries (
        query_id TEXT PRIMARY KEY,
        color TEXT,
        clothing_type TEXT,
        size TEXT,
        catalog_price REAL,
        channel TEXT,
        original_price REAL
    );                  
    """)

    cursor.execute("""
        CREATE TABLE results (
            results_id TEXT PRIMARY KEY,
            query_id TEXT,
            profit_margin REAL,
            quantity INT,
            item_total INT,
            FOREIGN KEY(query_id) REFERENCES queries(query_id)
    );
    """)
    
    connection.commit()
    connection.close()

if __name__ == "__main__":
    main()