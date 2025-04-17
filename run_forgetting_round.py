import time
import csv
import sys
import signal
import uuid
import pandas as pd
from datetime import datetime
from statistics import mean
from statistics import pstdev
from dep_df import *
from indep_df import *


# Auxiliary functions

def objective(dataset, dataset_prime_indices, queries, jaccard_sim=True):
    """The data forgetting objective f """
    data_array = dataset
    value = 0
    n_queries = len(queries)
    for q in range(n_queries): 
        prob_q = 1/n_queries
        answer_set_q_dataset = queries[q]
        partial_value = 0
        for d in answer_set_q_dataset:
            best_similarity_d = 0
            answer_set_q_dataset_prime = list(set(answer_set_q_dataset).intersection(dataset_prime_indices))
            for d_prime in answer_set_q_dataset_prime:
                if jaccard_sim:
                    similarity = jaccard(
                        data_array[d],
                        data_array[d_prime]
                    )
                else:
                    similarity = cosine_similarity(
                        [data_array[d]],
                        [data_array[d_prime]]
                    )[0][0]
                    similarity = (similarity + 1) / 2
                if similarity > best_similarity_d:
                    best_similarity_d = similarity
            partial_value += (1 / len(answer_set_q_dataset)) * best_similarity_d
        value += prob_q * partial_value
    return value


def compute_pairwise_sims(points, dataset, jaccard_sim):
    """
       This function computes the pairwise similarity between all 'points' in 'dataset' according 
       to the Jaccard or the Cosine similarity
    """
    n = len(points)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            d_i = dataset.loc[points[i]].values.flatten().tolist()
            d_j = dataset.loc[points[j]].values.flatten().tolist()
            if jaccard_sim:
                s = jaccard(d_i, d_j)
                if s != 0:
                    sims.append(s)
            else:
                s = cosine_similarity([d_i], [d_j])[0][0]
                s = (s+1)/2
                if s != 0:
                    sims.append(s)
    return sims


def execute_computations(dataset, n_rows, n_queries, queries, prob_queries, budget, results_filename, jaccard_sim=True, n_iterations=10000, only_time=False):
    """
        This function executes the search for D*\subseteq D 
        for the DepDF and IndepDF algorithms. Both time and quality
        of the solution (i.e., f(D*)) are measured. 
    """

    def handler():
        raise TimeoutError("Computation exceeded the time limit.")

    def run_with_timeout(func, *args):
        # Run the algorithm for a maximum of 3 days
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(3 * 24 * 60 * 60) 
        try:
            lasttime = time.time()
            result = func(*args)
            currtime = time.time()
            elapsed_time = (currtime - lasttime)
            return result, elapsed_time
        except TimeoutError as e:
            print(e)
            return None, -1
        finally:
            signal.alarm(0) 

    def print_results(method_name, result, elapsed_time):
        # Print and return the quality and runtime of the algorithm
        if result is not None and not only_time:
            utility = objective(dataset, result, queries, jaccard_sim)
            print(f"{method_name} utility: {utility}")
        else:
            utility = -1
        print(f"{method_name} time: {elapsed_time}")
        return elapsed_time, utility

    print(f'Tuples: {n_rows}, Queries: {n_queries}, Budget: {budget}')

    if not only_time:
        # Run each one of the algorithms 
        # INDEP_DF
        answer_indep_df, indep_df_time = run_with_timeout(indep_df, n_rows, budget, queries, n_queries)
        indep_df_time, indep_df_utility = print_results("Indep_df", answer_indep_df, indep_df_time)

        # DEP_DF
        def dep_df_computation():
            # Fetch the continuous solution x^* via SCG^2
            x_star = scg_squared(n_rows, budget, queries, prob_queries, n_queries, dataset, T=n_iterations, K=1, jaccard_sim=jaccard_sim)
            # Round the solution to a discrete one via randomized pipage rounding
            discrete_solution = pipage_rounding(x_star, n_rows, budget)
            return discrete_solution
        answer_dep_df, dep_df_time = run_with_timeout(dep_df_computation)
        dep_df_time, dep_df_utility = print_results("Dep_df", answer_dep_df, dep_df_time)
    else:
        # Run each one of the algorithms 
        # INDEP_DF
        print("Begin to run IndepDF")
        _, indep_df_time = run_with_timeout(indep_df, n_rows, budget, queries, n_queries)
        #_, indep_df_time = run_with_timeout(indep_df_parallel, n_rows, budget, queries, n_queries)

        # DEP_DF
        def dep_df_computation():
            x_star = scg_squared(n_rows, budget, queries, prob_queries, n_queries, dataset, T=n_iterations, K=1, jaccard_sim=jaccard_sim)
            return pipage_rounding(x_star, n_rows, budget)
        
        print("Begin to run DepDF")
        _, dep_df_time = run_with_timeout(dep_df_computation)
   
   # Write the results into a file
    with open(results_filename, 'w') as file:
        if not only_time:
            file.write(f'Algorithm, Time, Utility\n')
            file.write(f"INDEP_DF, {indep_df_time}, {indep_df_utility}\n")
            file.write(f"DEP_DF, {dep_df_time}, {dep_df_utility}\n")
        else:
            file.write(f'Algorithm, Time\n')
            file.write(f"INDEP_DF, {indep_df_time}\n")
            file.write(f"DEP_DF, {dep_df_time}\n")
        

def read_data(dataset_choice, percentage_of_db, percentage_of_ql, budget, n_iterations,av_stdevs_calculation, only_time):
    """
        This function allows you to read your dataset of choice 
    """

    dataset_paths = ["sample_data/real/flight data/flights.csv",
                    "sample_data/real/open photo data/photos.csv",
                    "sample_data/real/Wikidata/wikidata.csv",
                    "sample_data/synthetic/data_1000000_100000.parquet",
                    "sample_data/synthetic/data_1000000_1000000.parquet",
                    "sample_data/synthetic/data_1000000_10000000.parquet",
                    "sample_data/synthetic/data_5000000_10000000.parquet",
                    "sample_data/synthetic/data_10000000_10000000.parquet"]
    
    querylogs_paths = ["sample_data/real/flight data/flights_queries_ordered.csv",
                    "sample_data/real/open photo data/photos_queries_less100_ordered.csv",
                    "sample_data/real/Wikidata/wikiqueries_final.csv",
                    "sample_data/synthetic/queries_1000000_100000.parquet",
                    "sample_data/synthetic/queries_1000000_1000000.parquet",
                    "sample_data/synthetic/queries_1000000_10000000.parquet",
                    "sample_data/synthetic/queries_5000000_10000000.parquet",
                    "sample_data/synthetic/queries_10000000_10000000.parquet"]

    def load_data(dataset_choice):
        # Real datasets
        if dataset_choice == 'flight':
            dataset = pd.read_csv(dataset_paths[0], delimiter=",")
            n_queries = 37
            querylog_path = querylogs_paths[0]
        elif dataset_choice == 'photo':
            dataset = pd.read_csv(dataset_paths[1], delimiter=",", header=None)
            n_queries = 443
            querylog_path = querylogs_paths[1]
        elif dataset_choice == 'wiki':
            dataset = pd.read_csv(dataset_paths[2], delimiter=",", header=None)
            dataset.columns = ["id"] + [f'Value{i}' for i in range(1, 385)]
            dataset.drop('id', inplace=True, axis=1)
            n_queries = 14081
            querylog_path = querylogs_paths[2]
        
        # Synthetic datasets
        elif dataset_choice == 'S_1M_100K':
            dataset = pd.read_parquet(dataset_paths[3])
            dataset = dataset[1:].reset_index(drop=True)
            n_queries = 100000
            querylog_path = querylogs_paths[3]
        elif dataset_choice == 'S_1M_1M':
            dataset = pd.read_parquet(dataset_paths[4])
            dataset = dataset[1:].reset_index(drop=True)
            n_queries = 1000000
            querylog_path = querylogs_paths[4]
        elif dataset_choice == 'S_1M_10M':
            dataset = pd.read_parquet(dataset_paths[5])
            dataset = dataset[1:].reset_index(drop=True)
            n_queries = 10000000
            querylog_path = querylogs_paths[5]
        elif dataset_choice == 'S_5M_10M':
            dataset = pd.read_parquet(dataset_paths[6])
            dataset = dataset[1:].reset_index(drop=True)
            n_queries = 10000000
            querylog_path = querylogs_paths[6]
        elif dataset_choice == 'S_10M_10M':
            dataset = pd.read_parquet(dataset_paths[7])
            dataset = dataset[1:].reset_index(drop=True)
            n_queries = 10000000
            querylog_path = querylogs_paths[7]
        return dataset, n_queries, querylog_path

    # Load the data
    dataset, n_queries, querylog_path = load_data(dataset_choice)
    
    # Select the corresponding percentage of the database
    n_rows = dataset.shape[0]
    
    # Calculate the budget
    if budget < 1:
        budget = int(round(n_rows * budget))
    else:
        budget = int(budget)
    
    # Select the queries

    if dataset_choice in ['flight', 'photo', 'wiki']:
        queries = {}
        n_queries = int(round(n_queries * percentage_of_ql))
        query_list = []
        with open(querylog_path, 'r') as file:
            csvreader = csv.reader(file)
            i = 0
            for row in csvreader:
                if row == []:
                    continue
                elif i < n_queries:
                    answer_s = list(set([int(w) for w in row]).intersection(set([j for j in range(n_rows)])))
                    if answer_s == []:
                        i = i + 1
                        continue
                    else:
                        query_list.append(answer_s)
                        i = i + 1
                else:
                    break
        file.close()
        n_queries = len(query_list)
        prob_queries = [1 / n_queries] * n_queries
        for i in range(len(query_list)):
             queries[i] = query_list[i]
    
    else: 
        query_df = pd.read_parquet(querylog_path)
        query_df = query_df.reset_index(drop=True)
        queries = dict(zip(query_df.index, (query_df.values.tolist())))
        prob_queries = None

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    unique_id = uuid.uuid4()
    results_filename = f"sample_data/real/results_{dataset_choice}%db{percentage_of_db}_%ql{percentage_of_ql}_budget{budget}_iterations{n_iterations}_av{av_stdevs_calculation}_onlytime{only_time}_{timestamp}_{unique_id}"
    return dataset, n_rows, n_queries, queries, prob_queries, budget, results_filename


def main():
    # Execute a data forgetting round

    # First step: Get command line arguments
    dataset_choice = sys.argv[1]  # dataset_choice in [flight, photo, wiki] or the synthetic ones
    percentage_of_db = float(sys.argv[2])  # percentage of the database
    percentage_of_ql = float(sys.argv[3])  # percentage of the query-log
    budget = float(sys.argv[4])  # budget
    n_iterations = int(sys.argv[5])  # number of iterations of DepDF
    av_stdevs_calculation = bool(int(sys.argv[6]))  # 0/1
    only_time = bool(int(sys.argv[7]))  # 0/1

    # Depending on the dataset of choice set the appropiate simmilarity metric 
    if dataset_choice == 'flight':
        jaccard_sim = True
    else:
        jaccard_sim = False
    
    # Second step: Load the data into memory
    dataset, n_rows, n_queries, queries, prob_queries, budget, results_filename = read_data(dataset_choice,
        percentage_of_db, percentage_of_ql, budget, n_iterations, av_stdevs_calculation, only_time)

    # Third step: Compute the average diversity of the answer sets if av_stdevs_calculation = True
    if av_stdevs_calculation:
        stdevs = []
        for i in range(n_queries):
            print(f"iteration {i}/{n_queries}")
            query = list(set(queries[i][0]))
            if len(query) < 2:
                stdevs.append(0)
            else:
                stdevs.append(pstdev(compute_pairwise_sims(query, dataset, jaccard_sim=jaccard_sim)))
        print(f"The average standard deviation is: {mean(stdevs)}")
        avstdevs = mean(stdevs)
    
    # Fourth step: Execute the computations
    # Convert data into np.array
    dataset = dataset.values

    execute_computations(dataset, n_rows, n_queries, queries, prob_queries, budget, results_filename,
                         jaccard_sim=jaccard_sim, n_iterations=n_iterations, only_time=only_time)
    
    # Write into the output file the average diversity of the answer sets if av_stdevs_calculation = True
    if av_stdevs_calculation:
        f = open(f'{results_filename}', 'a')
        f.write(f"\nThe average standard deviation is: {avstdevs}")
        f.close()


if __name__ == '__main__':
    main()
