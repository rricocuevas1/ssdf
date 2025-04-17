import random
import csv
import os
import dask.dataframe as dd
import numpy as np


def convert_file_to_parquet(directory_path, filename):
    csv_filepath = os.path.join(directory_path, filename)
    print(csv_filepath)
    parquet_filename = os.path.splitext(filename)[0] + ".parquet"
    parquet_filepath = os.path.join(directory_path, parquet_filename)

    ddf = dd.read_csv(csv_filepath, header=None)
    ddf.columns = [str(col) for col in ddf.columns]
    ddf.to_parquet(parquet_filepath)

    print(f"Successfully converted '{csv_filepath}' to single Parquet file '{parquet_filepath}'")


def generate_synthetic_data(n_rows, n_queries):
    n_tuples_per_query = 10
    n_components_per_tuple = 100
    
    # Generate datasets
    with open(f"sample_data/synthetic/data_{n_rows}_{n_queries}.csv", 'w', newline='') as file:
        counter = 0
        writer = csv.writer(file)
        writer.writerow([f'c{i}' for i in range(n_components_per_tuple)]) # 100 columns
        for i in range(n_rows//n_tuples_per_query):
            for j in range(n_tuples_per_query):
                random.seed(i * j + i + j)
                writer.writerow([counter] + [random.random() for _ in range(n_components_per_tuple - 1)])
            counter = counter + 1
    file.close()
    print(f"Generated: sample_data/synthetic/data_{n_rows}_{n_queries}.csv")
    convert_file_to_parquet(os.getcwd()+"/sample_data/synthetic/", f"data_{n_rows}_{n_queries}.csv")
    
    # Generate query_logs
    with open(f"sample_data/synthetic/queries_{n_rows}_{n_queries}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(n_queries):
            random.seed(i)
            counter = random.choice(range(0, n_rows, 10))
            writer.writerow([i for i in range(counter, counter + n_tuples_per_query)])
    file.close()
    print(f"Generated: sample_data/synthetic/queries_{n_rows}_{n_queries}.csv")
    convert_file_to_parquet(os.getcwd()+"/sample_data/synthetic/", f"queries_{n_rows}_{n_queries}.csv")



os.mkdir(os.getcwd()+"/sample_data/synthetic/")
# Synthetic data (Generate datasets and logs)
#generate_synthetic_data(n_rows=1000000, n_queries=100000)
#generate_synthetic_data(n_rows=1000000, n_queries=1000000)
#generate_synthetic_data(n_rows=1000000, n_queries=10000000)
#generate_synthetic_data(n_rows=5000000, n_queries=10000000)
#generate_synthetic_data(n_rows=10000000, n_queries=10000000)
generate_synthetic_data(n_rows=10, n_queries=10)




