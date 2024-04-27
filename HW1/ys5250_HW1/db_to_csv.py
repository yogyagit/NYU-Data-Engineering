import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
import sqlite3

# Constants
SQL_DATABASE_PATH = 'my_sql.db'
MONGO_DB_URL = "mongodb://localhost:27017/"
MONGO_DB_NAME = "DE"
MONGO_COLLECTION_NAME = "fx"
CSV_SQL_PATH = 'csv_sql.csv'
CSV_MONGODB_PATH = 'csv_mongodb.csv'

# Function to export data from SQLite to CSV
def export_sql_to_csv():
    engine = create_engine(f'sqlite:///{SQL_DATABASE_PATH}')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    data = session.query(ForexData).all()
    df = pd.DataFrame([(entry.fx_timestamp, entry.pair, entry.rate, entry.entry_timestamp) for entry in data], columns=['FX_Timestamp', 'Pair', 'FX_Rate', 'Entry_Timestamp'])
    df.to_csv(CSV_SQL_PATH, index=False)
    
    session.close()

# Function to export data from MongoDB to CSV
def export_mongodb_to_csv():
    client = MongoClient(MONGO_DB_URL)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    data = list(collection.find({}, {'_id': 0}))
    df = pd.DataFrame(data)
    df.to_csv(CSV_MONGODB_PATH, index=False)

    client.close()

# Main function
def main():
    export_sql_to_csv()
    export_mongodb_to_csv()
    print("Data exported to CSV files.")

if __name__ == "__main__":
    main()
