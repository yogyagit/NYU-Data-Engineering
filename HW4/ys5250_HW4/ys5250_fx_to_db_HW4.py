import time
import pymongo
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
from pycaret.regression import setup, compare_models, pull
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Float, String, Integer, DateTime, func

forexList = [("EUR", "USD"), ("USD", "GBP"), ("GBP", "CAD")]
# Polygon REST client setup
client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")
Base = declarative_base()

# Function to fetch forex data from Polygon API
def fetch_forex_data(fx_pair):
    base_currency, quote_currency = fx_pair
    res = client.get_real_time_currency_conversion(base_currency, quote_currency, precision=7)
    fx_timestamp = datetime.utcfromtimestamp(res.last.timestamp / 1000.0)
    rate = res.converted
    pair = f"{base_currency},{quote_currency}"
    return pair, fx_timestamp, rate

# MongoDB setup
client_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
database_name = "MGGYDE_Spring24"
collection_1 = "HW4_EURUSD"
collection_2 = "HW4_USDGBP"
collection_3 = "HW4_GBPCAD"

# Forex data model for SQL
class ForexDataSQL_EURUSD(Base):
    __tablename__ = 'forex_data_EURUSD'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)  # This is the timestamp when the data is stored
    first_fx_timestamp = Column(DateTime)  # First fx timestamp of the 6-minute interval
    max_value = Column(Float)
    min_value = Column(Float)
    mean_value = Column(Float)
    vol_value = Column(Float)
    fd_value = Column(Float)
    corr_1 = Column(Float)
    corr_2 = Column(Float)

class ForexDataSQL_USDGBP(Base):
    __tablename__ = 'forex_data_USDGBP'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)  # This is the timestamp when the data is stored
    first_fx_timestamp = Column(DateTime)  # First fx timestamp of the 6-minute interval
    max_value = Column(Float)
    min_value = Column(Float)
    mean_value = Column(Float)
    vol_value = Column(Float)
    fd_value = Column(Float)
    corr_1 = Column(Float)
    corr_2 = Column(Float)

class ForexDataSQL_GBPCAD(Base):
    __tablename__ = 'forex_data_GBPCAD'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)  # This is the timestamp when the data is stored
    first_fx_timestamp = Column(DateTime)  # First fx timestamp of the 6-minute interval
    max_value = Column(Float)
    min_value = Column(Float)
    mean_value = Column(Float)
    vol_value = Column(Float)
    fd_value = Column(Float)
    corr_1 = Column(Float)
    corr_2 = Column(Float)


# Function to insert Forex data into MongoDB and calculate statistics
def insert_forex_data_and_calculate_stats(collection_name, pair, fx_timestamp, rate):
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
def store_statistics_in_sql(curr, session, timestamp, first_fx_timestamp, max_val, min_val, mean_val, vol_val, fd_val, corr_1, corr_2):
    if curr == 'EUR':    
        forex_entry_sql = ForexDataSQL_EURUSD(
            timestamp=timestamp,
            first_fx_timestamp=first_fx_timestamp,
            max_value=max_val,
            min_value=min_val,
            mean_value=mean_val,
            vol_value=vol_val,
            fd_value=fd_val,
            corr_1= corr_1,
            corr_2= corr_2
        )
        session.add(forex_entry_sql)
        session.commit()
    elif curr == 'USD':
        forex_entry_sql = ForexDataSQL_USDGBP(
            timestamp=timestamp,
            first_fx_timestamp=first_fx_timestamp,
            max_value=max_val,
            min_value=min_val,
            mean_value=mean_val,
            vol_value=vol_val,
            fd_value=fd_val,
            corr_1= corr_1,
            corr_2= corr_2
        )
        session.add(forex_entry_sql)
        session.commit()
    elif curr == 'GBP':
        forex_entry_sql = ForexDataSQL_GBPCAD(
            timestamp=timestamp,
            first_fx_timestamp=first_fx_timestamp,
            max_value=max_val,
            min_value=min_val,
            mean_value=mean_val,
            vol_value=vol_val,
            fd_value=fd_val,
            corr_1= corr_1,
            corr_2= corr_2
        )
        session.add(forex_entry_sql)
        session.commit()

def fetch_last_10_mean_values(session, table):
    query_result = session.query(table.mean_value).order_by(table.id.desc()).limit(10).all()
    return [result.mean_value for result in query_result]

def run_pycaret_experiment(session, table):
    # Fetch all data from the table
    query_result = session.query(table).all()
    data = pd.DataFrame([(row.timestamp, row.first_fx_timestamp, row.max_value, row.min_value, row.mean_value, row.vol_value, row.fd_value) for row in query_result],
                        columns=['timestamp', 'first_fx_timestamp', 'max_value', 'min_value', 'mean_value', 'vol_value', 'fd_value'])

    # Setup PyCaret environment
    setup(data, target='mean_value', verbose=False)

    # Compare baseline models and select the best one based on MAE
    best_model = compare_models(sort='MAE', n_select=1)

    comparison_results = pull()

    best_model_mae = comparison_results.iloc[0]['MAE']

    # Return the MAE of the best model
    return best_model_mae

# Main
def main():
    # SQLAlchemy setup
    SQL_DATABASE_PATH = 'ys5250_my_sql_HW4.db'
    SQL_ENGINE = create_engine(f'sqlite:///{SQL_DATABASE_PATH}')
    Session = sessionmaker(bind=SQL_ENGINE)
    Base.metadata.create_all(SQL_ENGINE)
    session = Session()

    keltner_upper_prev_EURUSD = []
    keltner_lower_prev_EURUSD = []

    keltner_upper_prev_USDGBP = []
    keltner_lower_prev_USDGBP = []

    keltner_upper_prev_GBPCAD = []
    keltner_lower_prev_GBPCAD = []

    for i in range(1, 51):  
        print(f"Processing cycle {i}/50")
        data_EURUSD = []
        data_USDGBP = []
        data_GBPCAD = []
        first_fx_timestamp_EURUSD = None 
        first_fx_timestamp_USDGBP = None
        first_fx_timestamp_GBPCAD = None

        results_df = pd.DataFrame(columns=['Cycle', 'Pair', 'Classification'])

        import os

        if i in [20, 30, 40, 50]:
            print(f"Running PyCaret experiments for cycle {i}")
            mae_eurusd = run_pycaret_experiment(session, ForexDataSQL_EURUSD)
            mae_usdgbp = run_pycaret_experiment(session, ForexDataSQL_USDGBP)
            mae_gbpcad = run_pycaret_experiment(session, ForexDataSQL_GBPCAD)

            results = {
                'EURUSD': mae_eurusd,
                'USDGBP': mae_usdgbp,
                'GBPCAD': mae_gbpcad
            }

            # Sort results by MAE to determine forecastability
            sorted_results = sorted(results.items(), key=lambda x: x[1])
            forecastability = {
                sorted_results[0][0]: 'FORECASTABLE',
                sorted_results[1][0]: 'UNDEFINED',
                sorted_results[2][0]: 'NON FORECASTABLE'
            }

            # Append results to DataFrame including MAE values
            for pair, classification in forecastability.items():
                new_row = pd.DataFrame({
                    'Cycle': [i],
                    'Pair': [pair],
                    'MAE': [results[pair]],  # Include the MAE value from the results
                    'Classification': [classification]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

            # Check if the file exists to determine if headers should be written
            file_exists = os.path.isfile('classification_results.csv')

            # Save the DataFrame to CSV, append if file exists, write header only if file does not exist
            results_df.to_csv('classification_results_final.csv', mode='a', header=not file_exists, index=False)


        for m in range(360):  # Fetch data every second for 6 minutes
            start = time.time()
            for baseCurrency, quoteCurrency in forexList:
                pair, fx_timestamp, rate = fetch_forex_data((baseCurrency, quoteCurrency))

                if baseCurrency == "EUR" and first_fx_timestamp_EURUSD is None:
                    first_fx_timestamp_EURUSD = fx_timestamp
                elif baseCurrency == "USD" and first_fx_timestamp_USDGBP is None:
                    first_fx_timestamp_USDGBP = fx_timestamp
                elif baseCurrency == "GBP" and first_fx_timestamp_GBPCAD is None:
                    first_fx_timestamp_GBPCAD = fx_timestamp

                print(f"pair: {pair}, fx_timestamp: {fx_timestamp}, rate = {rate}")
                if baseCurrency == "EUR":
                    max_val_EURUSD, min_val_EURUSD, mean_val_EURUSD, vol_val_EURUSD = insert_forex_data_and_calculate_stats(collection_1, pair, fx_timestamp, rate)
                    data_EURUSD.append(rate)
                elif baseCurrency == "USD":
                    max_val_USDGBP, min_val_USDGBP, mean_val_USDGBP, vol_val_USDGBP = insert_forex_data_and_calculate_stats(collection_2, pair, fx_timestamp, rate)
                    data_USDGBP.append(rate)
                elif baseCurrency == "GBP":
                    max_val_GBPCAD, min_val_GBPCAD, mean_val_GBPCAD, vol_val_GBPCAD = insert_forex_data_and_calculate_stats(collection_3, pair, fx_timestamp, rate)
                    data_GBPCAD.append(rate)
        
            end = time.time()
            elapsed = end - start 
            if elapsed < 1:
                time.sleep(1 - elapsed)
            else:
                print("Warning: iteration took more than 1 sec")
        
        if i > 10:
            eurusd_mean_values = fetch_last_10_mean_values(session, ForexDataSQL_EURUSD)
            usdgbp_mean_values = fetch_last_10_mean_values(session, ForexDataSQL_USDGBP)
            gbpcad_mean_values = fetch_last_10_mean_values(session, ForexDataSQL_GBPCAD)

            df = pd.DataFrame({
                'EURUSD': eurusd_mean_values,
                'USDGBP': usdgbp_mean_values,
                'GBPCAD': gbpcad_mean_values
            })

            # Calculate correlation matrix
            correlation_matrix = df.corr()

            # Extract specific correlation values
            corr_AB = correlation_matrix.loc['EURUSD', 'USDGBP']
            corr_BC = correlation_matrix.loc['USDGBP', 'GBPCAD']
            corr_AC = correlation_matrix.loc['EURUSD', 'GBPCAD']
            print('corr_AB:', corr_AB)
            print('corr_AB:', corr_BC)
            print('corr_AB:', corr_AC)

        data = [data_EURUSD, data_USDGBP, data_GBPCAD]
        # After 6 minutes, calculate Keltner Bands and FD, then store all data in SQL database
        for d in data:
            if d == data_EURUSD:
                if i > 1:  
                    fd_val = calc_fd(data_EURUSD, keltner_upper_prev_EURUSD, keltner_lower_prev_EURUSD)
                else:
                    fd_val = 0.0
            if d == data_USDGBP:
                if i > 1:  
                    fd_val = calc_fd(data_USDGBP, keltner_upper_prev_USDGBP, keltner_lower_prev_USDGBP)
                else:
                    fd_val = 0.0
            if d == data_GBPCAD:
                if i > 1:  
                    fd_val = calc_fd(data_GBPCAD, keltner_upper_prev_GBPCAD, keltner_lower_prev_GBPCAD)
                else:
                    fd_val = 0.0

            if d == data_EURUSD:
                if i <= 10:
                    store_statistics_in_sql('EUR',session, datetime.now(), first_fx_timestamp_EURUSD, max_val_EURUSD, min_val_EURUSD, mean_val_EURUSD, vol_val_EURUSD, fd_val, 0, 0)   
                else:
                    store_statistics_in_sql('EUR', session, datetime.now(), first_fx_timestamp_EURUSD, max_val_EURUSD, min_val_EURUSD, mean_val_EURUSD, vol_val_EURUSD, fd_val, corr_AB, corr_AC)   
                keltner_upper_prev_EURUSD, keltner_lower_prev_EURUSD = calculate_keltner_bands(mean_val_EURUSD, vol_val_EURUSD)
                print(f"Max Value: {max_val_EURUSD}, Min Value: {min_val_EURUSD}, Mean Value: {mean_val_EURUSD}, VOL Value: {vol_val_EURUSD}, Fractal Dimension: {fd_val}")
            if d == data_USDGBP:
                if i <= 10:
                    store_statistics_in_sql('USD', session, datetime.now(), first_fx_timestamp_USDGBP, max_val_USDGBP, min_val_USDGBP, mean_val_USDGBP, vol_val_USDGBP, fd_val, 0, 0)
                else:
                    store_statistics_in_sql('USD', session, datetime.now(), first_fx_timestamp_USDGBP, max_val_USDGBP, min_val_USDGBP, mean_val_USDGBP, vol_val_USDGBP, fd_val, corr_AB, corr_BC)   
                keltner_upper_prev_USDGBP, keltner_lower_prev_USDGBP = calculate_keltner_bands(mean_val_USDGBP, vol_val_USDGBP)
                print(f"Max Value: {max_val_USDGBP}, Min Value: {min_val_USDGBP}, Mean Value: {mean_val_USDGBP}, VOL Value: {vol_val_USDGBP}, Fractal Dimension: {fd_val}")
            if d == data_GBPCAD:
                if i <= 10:
                    store_statistics_in_sql('GBP', session, datetime.now(), first_fx_timestamp_GBPCAD, max_val_GBPCAD, min_val_GBPCAD, mean_val_GBPCAD, vol_val_GBPCAD, fd_val, 0, 0)  
                else:
                    store_statistics_in_sql('GBP', session, datetime.now(), first_fx_timestamp_GBPCAD, max_val_GBPCAD, min_val_GBPCAD, mean_val_GBPCAD, vol_val_GBPCAD, fd_val, corr_AC, corr_BC)  
                keltner_upper_prev_GBPCAD, keltner_lower_prev_GBPCAD = calculate_keltner_bands(mean_val_GBPCAD, vol_val_GBPCAD)
                print(f"Max Value: {max_val_GBPCAD}, Min Value: {min_val_GBPCAD}, Mean Value: {mean_val_GBPCAD}, VOL Value: {vol_val_GBPCAD}, Fractal Dimension: {fd_val}")
            
        # Clear MongoDB collection for next 6 minutes
        db = client_mongo[database_name]
        collections = [collection_1, collection_2, collection_3]
        for collection_name in collections:
            collection = db[collection_name]
            collection.delete_many({})

    session.close()
    

if __name__ == "__main__":
    main()
