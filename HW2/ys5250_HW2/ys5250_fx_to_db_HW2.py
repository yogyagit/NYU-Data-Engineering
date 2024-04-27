import time
from polygon import RESTClient
from datetime import datetime, timedelta
import pymongo
from sqlalchemy import create_engine, Column, Float, String, Integer, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Polygon REST client setup
client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")
Base = declarative_base()

# Function to fetch forex data from Polygon API
def fetch_forex_data():
    fx_pair = ("EUR", "USD")  
    base_currency, quote_currency = fx_pair
    res = client.get_real_time_currency_conversion(base_currency, quote_currency, precision=7)
    fx_timestamp = datetime.utcfromtimestamp(res.last.timestamp / 1000.0)
    rate = res.converted
    pair = f"{base_currency},{quote_currency}"
    return pair, fx_timestamp, rate

# MongoDB setup
client_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
database_name = "MGGYDE_Spring24"
collection_name = "HW2_forex_aggregates"

# Forex data model for SQL
class ForexDataSQL(Base):
    __tablename__ = 'forex_data'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)  # This is the timestamp when the data is stored
    first_fx_timestamp = Column(DateTime)  # First fx timestamp of the 6-minute interval
    max_value = Column(Float)
    min_value = Column(Float)
    mean_value = Column(Float)
    vol_value = Column(Float)
    fd_value = Column(Float)


# Function to insert Forex data into MongoDB and calculate statistics
def insert_forex_data_and_calculate_stats(pair, fx_timestamp, rate):
    db = client_mongo[database_name]
    collection = db[collection_name]
    entry = {
        "pair": pair,
        "fx_timestamp": fx_timestamp,
        "rate": rate,
        "entry_timestamp": datetime.now()
    }
    collection.insert_one(entry)

    # Calculate statistics
    data = [entry['rate'] for entry in collection.find()]
    max_val = max(data)
    min_val = min(data)
    mean_val = sum(data) / len(data)
    vol_val = (max_val - min_val) / mean_val
    
    return max_val, min_val, mean_val, vol_val

# Function to calculate Keltner Bands
def calculate_keltner_bands(mean_val, vol_val):
    keltner_upper = [mean_val + n * 0.025 * vol_val for n in range(1, 101)]
    keltner_lower = [mean_val - n * 0.025 * vol_val for n in range(1, 101)]
    return keltner_upper, keltner_lower

# Function to calculate fractal dimension
def calc_fd(prices, keltner_upper, keltner_lower):
    if not prices or not keltner_upper or not keltner_lower:
        return None

    crossings = 0

    for i in range(1, len(prices)):
        prev_price, curr_price = prices[i-1], prices[i]

        # Iterate through Keltner channels
        for j in range(len(keltner_upper)-1):
            upper_crossed_upward = prev_price <= keltner_upper[j] and curr_price > keltner_upper[j+1]
            upper_crossed_downward = prev_price > keltner_upper[j+1] and curr_price <= keltner_upper[j]
            
            lower_crossed_downward = prev_price >= keltner_lower[j] and curr_price < keltner_lower[j+1]
            lower_crossed_upward = prev_price < keltner_lower[j+1] and curr_price >= keltner_lower[j]

            # Count crossings
            crossings += upper_crossed_upward + upper_crossed_downward + lower_crossed_downward + lower_crossed_upward

    price_range = max(prices) - min(prices)
    if price_range == 0:
        return 0  

    fd = crossings / price_range
    return fd

# Fuction to store statistics in SQL db
def store_statistics_in_sql(session, timestamp, first_fx_timestamp, max_val, min_val, mean_val, vol_val, fd_val):
    forex_entry_sql = ForexDataSQL(
        timestamp=timestamp,
        first_fx_timestamp=first_fx_timestamp,
        max_value=max_val,
        min_value=min_val,
        mean_value=mean_val,
        vol_value=vol_val,
        fd_value=fd_val
    )
    session.add(forex_entry_sql)
    session.commit()

# Main
def main():
    # SQLAlchemy setup
    SQL_DATABASE_PATH = 'ys5250_my_sql_HW4.db'
    SQL_ENGINE = create_engine(f'sqlite:///{SQL_DATABASE_PATH}')
    Session = sessionmaker(bind=SQL_ENGINE)
    Base.metadata.create_all(SQL_ENGINE)
    session = Session()

    keltner_upper_prev = []
    keltner_lower_prev = []

    for i in range(1, 51):  
        print(f"Processing cycle {i}/50...")
        data = []
        first_fx_timestamp = None 

        for i in range(360):  # Fetch data every second for 6 minutes
            start = time.time()
            pair, fx_timestamp, rate = fetch_forex_data()

            if first_fx_timestamp is None:
                first_fx_timestamp = fx_timestamp

            print(f"pair: {pair}, fx_timestamp: {fx_timestamp}, rate = {rate}")
            max_val, min_val, mean_val, vol_val = insert_forex_data_and_calculate_stats(pair, fx_timestamp, rate)
            data.append(rate)

            end = time.time()
            elapsed = end - start 
            if elapsed < 1:
                time.sleep(1 - elapsed)
            else:
                print("Warning: iteration took more than 1 sec")
        
        # After 6 minutes, calculate Keltner Bands and FD, then store all data in SQL database
        if data:
            if i > 1:  
                fd_val = calc_fd(data, keltner_upper_prev, keltner_lower_prev)
            else:
                fd_val = 0.0

            print(f"Max Value: {max_val}, Min Value: {min_val}, Mean Value: {mean_val}, VOL Value: {vol_val}, Fractal Dimension: {fd_val}")
            store_statistics_in_sql(session, datetime.now(), first_fx_timestamp, max_val, min_val, mean_val, vol_val, fd_val)
            keltner_upper_prev, keltner_lower_prev = calculate_keltner_bands(mean_val, vol_val)
            
        # Clear MongoDB collection for next 6 minutes
        db = client_mongo[database_name]
        collection = db[collection_name]
        collection.delete_many({})
    session.close()
    

if __name__ == "__main__":
    main()
