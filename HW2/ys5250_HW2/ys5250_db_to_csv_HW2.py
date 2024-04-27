import sqlite3
import csv

db_path = 'ys5250_my_sql_HW2.db'

csv_file_path = 'ys5250_HW2.csv'

# SQL query to select data
sql_query = 'SELECT * FROM forex_data'  # Replace 'your_table_name' with the actual table name

conn = sqlite3.connect(db_path)
cursor = conn.cursor()


cursor.execute(sql_query)

# Fetch all rows from the query result
rows = cursor.fetchall()

# Get column headers
column_headers = [description[0] for description in cursor.description]

# Write to CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(column_headers)  # Write the headers first
    csv_writer.writerows(rows)  # Then write the data

conn.close()

print(f"Data exported successfully to {csv_file_path}")

