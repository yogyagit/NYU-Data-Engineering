import time
import pymongo
import pandas as pd
from polygon import RESTClient
from datetime import datetime
from pycaret.regression import setup, compare_models, pull, save_model

# Forex pairs to track
forexList = [("EUR", "USD"), ("GBP", "CHF"), ("USD", "CAD"), ("EUR", "CHF"), ("EUR", "CAD"),
               ("GBP", "EUR"), ("GBP", "USD"), ("GBP", "CAD"), ("USD", "CHF"), ("USD", "JPY")]

# Polygon REST client setup
client = RESTClient("beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq")

# Function to fetch forex data from Polygon API
def fetch_forex_data(fx_pair):
    base_currency, quote_currency = fx_pair
    res = client.get_real_time_currency_conversion(base_currency, quote_currency, precision=7)
    fx_timestamp = datetime.utcfromtimestamp(res.last.timestamp / 1000.0)
    if quote_currency == "JPY":
        rate = res.converted/1000
    else:
        rate = res.converted
    pair = f"{base_currency},{quote_currency}"
    return pair, fx_timestamp, rate


# MongoDB setup
client_mongo = pymongo.MongoClient("mongodb://localhost:27017/")
database_name = "MGGYDE_Spring24"
collection_EURUSD = "collection_EURUSD"
collection_GBPCHF = "collection_GBPCHF"
collection_USDCAD = "collection_USDCAD"
collection_EURCHF = "collection_EURCHF"
collection_EURCAD = "collection_EURCAD"
collection_GBPEUR = "collection_GBPEUR"
collection_GBPUSD = "collection_GBPUSD"
collection_GBPCAD = "collection_GBPCAD"
collection_USDCHF = "collection_USDCHF"
collection_USDJPY = "collection_USDJPY"


collection_EURUSD_stats = "collection_EURUSD_stats"
collection_GBPCHF_stats  = "collection_GBPCHF_stats"
collection_USDCAD_stats  = "collection_USDCAD_stats"
collection_EURCHF_stats  = "collection_EURCHF_stats"
collection_EURCAD_stats  = "collection_EURCAD_stats"
collection_GBPEUR_stats  = "collection_GBPEUR_stats"
collection_GBPUSD_stats  = "collection_GBPUSD_stats"
collection_GBPCAD_stats  = "collection_GBPCAD_stats"
collection_USDCHF_stats  = "collection_USDCHF_stats"
collection_USDJPY_stats = "collection_USDJPY_stats"
 
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

def insert_stats(collection_name, fx_timestamp,  max_val, min_val, mean_val, vol_val, fd_val, corr_EURUSD, corr_USDJPY):
    db = client_mongo[database_name]
    collection = db[collection_name]
    entry = {
        "fx_timestamp": fx_timestamp,
        "entry_timestamp": datetime.now(),
        "max_value": max_val,
        "min_value": min_val,
        "mean_value": mean_val,
        "vol_value": vol_val,
        "fd_value": fd_val,
        "corr_EURUSD": corr_EURUSD,
        "corr_USDJPY": corr_USDJPY

    }
    collection.insert_one(entry)

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

import pandas as pd
import pymongo
from pycaret.regression import setup, compare_models, pull, save_model

def run_pycaret_experiment():
    # Fetch all data from the MongoDB collection
    db = client_mongo[database_name]
    synth_collection = db["synth_pair_stats"]
    data = list(synth_collection.find())
    
    # Convert the data to a pandas DataFrame and drop the MongoDB ID
    df = pd.DataFrame(data)
    df.drop('_id', axis=1, inplace=True)
    
    # Setup PyCaret environment
    setup(df, target='mean_value', session_id=123, verbose=False)
    
    # Compare baseline models and select the best one based on MAE
    best_model = compare_models(sort='MAE', n_select=1)
    
    # Save the best model
    model_name = "best_model"
    save_model(best_model, model_name)
    
    # Pull the results to get MAE of the best model
    comparison_results = pull()
    best_model_mae = comparison_results.iloc[0]['MAE']
    
    print(f"Best model saved as {model_name} with MAE: {best_model_mae}")
    
    # Return the MAE of the best model
    return best_model_mae

# Define the collections for stats
collections_stats = {
    "EURUSD": "collection_EURUSD_stats",
    "GBPCHF": "collection_GBPCHF_stats",
    "USDCAD": "collection_USDCAD_stats",
    "EURCHF": "collection_EURCHF_stats",
    "EURCAD": "collection_EURCAD_stats",
    "GBPEUR": "collection_GBPEUR_stats",
    "GBPUSD": "collection_GBPUSD_stats",
    "GBPCAD": "collection_GBPCAD_stats",
    "USDCHF": "collection_USDCHF_stats",
    "USDJPY": "collection_USDJPY_stats"
}

import pandas as pd
import pymongo
from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_absolute_error

def calculate_mae_for_currency_pairs(model_name):
    db = client_mongo[database_name]
    mae_results = {}
    
    # Load the saved model
    model = load_model(model_name)
    
    for collection in collections_stats.keys():
        collection_name = collections_stats[collection]
        collection = db[collection_name]
        
        # Fetch the last 20 data points
        data_points = list(collection.find().sort('_id', -1).limit(20))
        df = pd.DataFrame(data_points)
        
        if not df.empty:
            # Assuming 'df' is your DataFrame and 'model' is your loaded prediction model

            # Drop the unnecessary columns
            df.drop(['_id', 'fx_timestamp', 'entry_timestamp'], axis=1, inplace=True)

            # Separate the target variable and features
            features = df.drop('mean_value', axis=1)
            target = df['mean_value']

            # Predict using the loaded model
            predictions = predict_model(model, data=features)

            # Add predictions to the DataFrame for comparison
            df['predicted_mean_value'] = predictions['prediction_label']

            # Calculate MAE between the actual and predicted values
            mae = mean_absolute_error(target, df['predicted_mean_value'])
            mae_results[collection] = mae
        else:
            mae_results[collection] = None  # No data available for calculation
    
    return mae_results

# Function to fetch the last 20 mean values from each collection
def fetch_last_20_mean_values(collection_name):
    db = client_mongo[database_name]
    collection = db[collection_name]
    data_points = collection.find().sort("entry_timestamp", -1).limit(20)
    return [data['mean_value'] for data in data_points]

# Function to fetch the last 20 entries from a collection
def fetch_last_20_entries(collection_name, lim):
    db = client_mongo[database_name]
    collection = db[collection_name]
    return list(collection.find().sort("entry_timestamp", -1).limit(lim))

def create_synth_pair(cycle):
    if cycle == 40:
        lim = 20
    else:
        lim = 30

    # Fetch data from each collection
    eurusd_data = fetch_last_20_entries(collections_stats["EURUSD"], lim)
    gbpchf_data = fetch_last_20_entries(collections_stats["GBPCHF"], lim)
    usdcad_data = fetch_last_20_entries(collections_stats["USDCAD"], lim)

    # Prepare the new collection for aggregated data
    synth_collection_name = "synth_pair_stats"
    db = client_mongo[database_name]

    # Check if the collection exists
    if synth_collection_name in db.list_collection_names():
        # Drop the collection if it exists
        db[synth_collection_name].drop()
        print(f"Dropped existing collection: {synth_collection_name}")

    # Create the collection again
    synth_collection = db[synth_collection_name]
    print(f"Created new collection: {synth_collection_name}")

    # Calculate averages and insert into new collection
    for i in range(lim):
        avg_max = (eurusd_data[i]['max_value'] + gbpchf_data[i]['max_value'] + usdcad_data[i]['max_value']) / 3
        avg_min = (eurusd_data[i]['min_value'] + gbpchf_data[i]['min_value'] + usdcad_data[i]['min_value']) / 3
        avg_mean = (eurusd_data[i]['mean_value'] + gbpchf_data[i]['mean_value'] + usdcad_data[i]['mean_value']) / 3
        avg_vol = (eurusd_data[i]['vol_value'] + gbpchf_data[i]['vol_value'] + usdcad_data[i]['vol_value']) / 3
        avg_fd = (eurusd_data[i]['fd_value'] + gbpchf_data[i]['fd_value'] + usdcad_data[i]['fd_value']) / 3

        # Inserting aggregated data into the new collection
        synth_collection.insert_one({
            "max_value": avg_max,
            "min_value": avg_min,
            "mean_value": avg_mean,
            "vol_value": avg_vol,
            "fd_value": avg_fd
        })

    print("Aggregated data has been successfully stored in the collection", synth_collection_name)

def extract_currency_pair_name(collection_name):
    # Extracts the currency pair name from the Collection string
    collection_prefix = "collection_"
    stats_suffix = "_stats"
    start = collection_name.find(collection_prefix) + len(collection_prefix)
    end = collection_name.rfind(stats_suffix)
    return collection_name[start:end]

def transform_mae_results(mae_results):
    transformed_results = {}
    for collection, value in mae_results.items():
        # Convert the Collection object to string
        collection_str = str(collection)
        # Extract the currency pair name
        currency_pair = extract_currency_pair_name(collection_str)
        # Map the extracted name to its corresponding value
        transformed_results[currency_pair] = value
    return transformed_results

# Main
def main():
    keltner_prev = {}
    for baseCurrency, quoteCurrency in forexList:
        pair_key = f"{baseCurrency}{quoteCurrency}"
        keltner_prev[pair_key] = {'upper': None, 'lower': None}

    for i in range(1, 51):  
        print(f"Processing cycle {i}/50")
        
        results_df = pd.DataFrame(columns=['Cycle', 'Pair', 'Classification'])

        import os

        if i in [40, 50]:
            print(f"Running PyCaret experiments for cycle {i}")

            create_synth_pair(i)
            run_pycaret_experiment()

            # Example usage
            model_name = 'best_model'  # Adjust based on your actual saved model's name
            mae_results = calculate_mae_for_currency_pairs(model_name)
            print(mae_results)

            transformed_results = transform_mae_results(mae_results)

            # Sort results by MAE to determine forecastability
            sorted_results = sorted(transformed_results.items(), key=lambda x: x[1])
            
            sorted_dict = {pair: value for pair, value in sorted_results}

            forecastability = {
                sorted_results[0][0]: 'FORECASTABLE',
                sorted_results[1][0]: 'FORECASTABLE',
                sorted_results[2][0]: 'FORECASTABLE',
                sorted_results[3][0]: 'UNDEFINED',
                sorted_results[4][0]: 'UNDEFINED',
                sorted_results[5][0]: 'UNDEFINED',
                sorted_results[6][0]: 'UNDEFINED',
                sorted_results[7][0]: 'NON FORECASTABLE',
                sorted_results[8][0]: 'NON FORECASTABLE',
                sorted_results[9][0]: 'NON FORECASTABLE'
            }

            # Append results to DataFrame including MAE values
            for pair, classification in forecastability.items():
                new_row = pd.DataFrame({
                    'Cycle': [i],
                    'Pair': [pair],
                    'MAE': [sorted_dict[pair]],  # Include the MAE value from the results
                    'Classification': [classification]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

            # Check if the file exists to determine if headers should be written
            file_exists = os.path.isfile('classification_results.csv')

            # Save the DataFrame to CSV, append if file exists, write header only if file does not exist
            results_df.to_csv('classification_results.csv', mode='a', header=not file_exists, index=False)

        data = {}
        stats = {}
        
        # Initialize the structures for each pair
        for baseCurrency, quoteCurrency in forexList:
            pair_key = f"{baseCurrency}{quoteCurrency}"
            data[pair_key] = []
            stats[pair_key] = {'max': None, 'min': None, 'mean': None, 'vol': None, 'fd_val': 0.0}

        for m in range(360):  # Fetch data every second for 6 minutes
            start = time.time()
            for baseCurrency, quoteCurrency in forexList:
                first_fx_timestamps = {}
                pair, fx_timestamp, rate = fetch_forex_data((baseCurrency, quoteCurrency))

                pair_key = f"{baseCurrency}{quoteCurrency}"
                collection = "collection_" + pair_key
                # If the pair key is not in the dictionary, add the timestamp
                if pair_key not in first_fx_timestamps:
                    first_fx_timestamps[pair_key] = fx_timestamp

                print(f"pair: {pair}, fx_timestamp: {fx_timestamp}, rate = {rate}")
                stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'] = insert_forex_data_and_calculate_stats(collection, pair, fx_timestamp, rate)
                data[pair_key].append(rate)
                
            end = time.time()
            elapsed = end - start 
            if elapsed < 1:
                time.sleep(1 - elapsed)
            else:
                print("Warning: iteration took more than 1 sec")
        
        if i > 20:
            # Fetch data and prepare DataFrame
            stats_data = {}
            for pair, coll_name in collections_stats.items():
                stats_data[pair] = fetch_last_20_mean_values(coll_name)

            df = pd.DataFrame(stats_data)

            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            # Organize correlations in a nested dictionary
            correlations = {}
            for pair in df.columns:
                correlations[pair] = {
                    "corr_EURUSD": correlation_matrix.at[pair, "EURUSD"],
                    "corr_USDJPY": correlation_matrix.at[pair, "USDJPY"]
                }

        # After 6 minutes, calculate Keltner Bands and FD, then store all data in SQL database
        for pair_key in data.keys():
            if i > 1:
                stats[pair_key]['fd_val'] = calc_fd(data[pair_key], keltner_prev[pair_key]['upper'], keltner_prev[pair_key]['lower'])
            else:
                stats[pair_key]['fd_val'] = 0.0

            collection = "collection_" + pair_key + "_stats"

            if i <= 20:
                insert_stats(collection, fx_timestamp, stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'], stats[pair_key]['fd_val'], 0, 0)
            else:
                insert_stats(collection, fx_timestamp, stats[pair_key]['max'], stats[pair_key]['min'], stats[pair_key]['mean'], stats[pair_key]['vol'], stats[pair_key]['fd_val'], correlations[pair_key]["corr_EURUSD"], correlations[pair_key]["corr_USDJPY"])
            keltner_prev[pair_key]['upper'], keltner_prev[pair_key]['lower'] = calculate_keltner_bands(stats[pair_key]['mean'], stats[pair_key]['vol'])

        # Clear MongoDB collection for next 6 minutes
        db = client_mongo[database_name]
        collections = [collection_EURUSD, collection_GBPCHF, collection_USDCAD, collection_EURCHF, collection_EURCAD, collection_GBPEUR, collection_GBPUSD, collection_GBPCAD, collection_USDCHF, collection_USDJPY]
        for collection_name in collections:
            collection = db[collection_name]
            collection.delete_many({})
    

if __name__ == "__main__":
    main()