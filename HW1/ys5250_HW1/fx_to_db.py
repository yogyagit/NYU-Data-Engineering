import pandas as pd
import time
from sqlalchemy import create_engine, Column, Float, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from polygon import RESTClient
from datetime import datetime
import pymongo

# Constants
sqlDatabasePath = 'my_sql.db'
csvOutputPath = 'fx_rates3.csv'
mongoDbUrl = "mongodb://localhost:27017/"
mongoDbName = "DE"
mongoCollectionName = "fx"
forexList = [("EUR", "USD"), ("USD", "GBP"), ("GBP", "CAD")]

# Initialize Polygon REST client
polygonClient = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")

# SQLAlchemy database setup
Base = declarative_base()

class ForexData(Base):
    __tablename__ = 'forex_data'
    id = Column(Integer, primary_key=True)
    pair = Column(String)
    fx_timestamp = Column(DateTime)
    rate = Column(Float)
    entry_timestamp = Column(DateTime, default=datetime.now)

    def __init__(self, pair, fx_timestamp, rate):
        self.pair = pair
        self.fx_timestamp = fx_timestamp
        self.rate = rate

# MongoDB connection setup
mongoClient = pymongo.MongoClient(mongoDbUrl)
mongoDb = mongoClient[mongoDbName]
mongoCollection = mongoDb[mongoCollectionName]

# SQLite session setup
def createSqlSession():
    engine = create_engine(f'sqlite:///{sqlDatabasePath}')
    Base.metadata.create_all(engine)  # Creates the database tables
    Session = sessionmaker(bind=engine)
    return Session()

# Function to fetch Forex data for a given currency pair
def fetchForexData(baseCurrency, quoteCurrency):
    try:
        res = polygonClient.get_real_time_currency_conversion(baseCurrency, quoteCurrency, precision=2)
        fxTimestamp = datetime.utcfromtimestamp(res.last.timestamp / 1000.0)
        rate = res.converted
        pair = f"{baseCurrency},{quoteCurrency}"
        return pair, fxTimestamp, rate
    except Exception as e:
        print(f"Error occurred while fetching data for {baseCurrency}-{quoteCurrency}: {e}")
        return None, None, None

# Function to store data in SQLite
def storeInSqlite(session, pair, fxTimestamp, rate):
    forexEntrySql = ForexData(pair=pair, fx_timestamp=fxTimestamp, rate=rate)
    session.add(forexEntrySql)

# Function to store data in MongoDB
def storeInMongoDB(pair, fxTimestamp, rate):
    document = {
        "pair": pair,
        "fx_timestamp": fxTimestamp,
        "rate": rate,
        "entry_timestamp": datetime.utcnow()
    }
    mongoCollection.insert_one(document)

# Main function to fetch and store Forex data
def fetchAndStoreForexData():
    totalIterations = 0
    iterationsExceeding1Sec = 0
    totalTimeExceeding1Sec = 0

    session = createSqlSession()

    startTime = time.time()
    while (time.time() - startTime) < 7200:  # Run for 2 hours
        startIterationTime = time.time()

        print("Fetching and storing forex data for each currency pair...")
        
        for baseCurrency, quoteCurrency in forexList:
            pair, fxTimestamp, rate = fetchForexData(baseCurrency, quoteCurrency)
            if pair is not None:
                storeInSqlite(session, pair, fxTimestamp, rate)
                storeInMongoDB(pair, fxTimestamp, rate)
                print(f"Data added to SQLite and MongoDB: Pair: {pair}, FX Timestamp: {fxTimestamp}, Rate: {rate}")

        session.commit()

        endIterationTime = time.time()
        iterationTime = endIterationTime - startIterationTime
        print(f"Iteration time: {iterationTime} seconds")
        
        totalIterations += 1
        if iterationTime > 1:
            iterationsExceeding1Sec += 1
            totalTimeExceeding1Sec += iterationTime - 1 
        
        print(f"Total number of iterations exceeding 1 sec till now: {iterationsExceeding1Sec}/{totalIterations}")
        print(f"Total time exceeding 1 sec: {totalTimeExceeding1Sec} seconds")

        if iterationTime < 1:
            time.sleep(1 - iterationTime)  # Adjust the waiting time to maintain approximately 1-second intervals

    session.close()
    print("Data collection complete.")

# Main function
def main():
    fetchAndStoreForexData()

if __name__ == "__main__":
    main()
