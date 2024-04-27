import pandas as pd
import time
import pdb
from sqlalchemy import create_engine, Column, Float, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from polygon import RESTClient
from datetime import datetime, timedelta


# Specify DB path
DATABASE_PATH = 'my_sql.db'
CSV_OUTPUT_PATH = 'fx_rates3.csv'

# Forex pairs to fetch data for
forex_list = [("EUR", "USD"), ("GBP", "USD"), ("JPY", "INR")]

# Initialize Polygon REST client
client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")

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

# Create SQLite engine and session
engine = create_engine(f'sqlite:///{DATABASE_PATH}')
Session = sessionmaker(bind=engine)

# Function to fetch and store forex data
def fetch_and_store_forex_data():
    with Session() as session:
        Base.metadata.create_all(engine)
        
        for fx_pair in forex_list:
            base_currency, quote_currency = fx_pair
            res = client.get_real_time_currency_conversion(base_currency, quote_currency, precision=2)
            fx_timestamp = datetime.utcfromtimestamp(res.last.timestamp / 1000.0)  # Convert milliseconds to UTC timestamp
            rate = res.converted
        
            pair = f"{base_currency},{quote_currency}"
            
            # Store forex data in the database
            forex_entry = ForexData(pair=pair, fx_timestamp=fx_timestamp, rate=rate)
            session.add(forex_entry)
            session.commit()

            print(f"{pair} - Price: {rate}")

# Run the code for 2 hours with data points every second
duration = timedelta(hours=2)
end_time = datetime.now() + duration
timer = datetime.now() + timedelta(minutes=10)

while datetime.now() < end_time:
    fetch_and_store_forex_data()
    time.sleep(1)  # Wait for 1 second before fetching data again
    
    if datetime.now() >= timer:
        print("Hello")
        timer = datetime.now() + timedelta(minutes=10)

# Export data to CSV
with Session() as session:
    data = session.query(ForexData).all()
    df = pd.DataFrame([(entry.fx_timestamp, entry.pair, entry.rate, entry.entry_timestamp) for entry in data], columns=['FX_Timestamp', 'Pair', 'FX_Rate', 'Entry_Timestamp'])
    df.to_csv(CSV_OUTPUT_PATH, index=False)
