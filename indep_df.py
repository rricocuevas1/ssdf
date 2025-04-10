import numpy as np
from itertools import islice
from multiprocessing import Pool, cpu_count


# Sequential IndepDF 
def compute_weight(n_rows, queries, n_queries):
    """
        Sequential weight computation
    """
    weight = np.zeros(n_rows)
    for j in range(n_queries):
        answer_set = queries[j]
        query_weight = 1/n_queries
        per_element_weight = query_weight / len(answer_set)
        np.add.at(weight, answer_set, per_element_weight)
    return weight

def indep_df(n_rows, budget, queries, n_queries):
    """
        IndepDF algorithm: Weight computations computed sequentially
    """
    values = compute_weight(n_rows, queries, n_queries)
    top_indices = np.argpartition(values, -budget)[-budget:]
    return top_indices.tolist()


# Parallel IndepDF
def chunked_iterable(iterable, chunk_size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, chunk_size))
            if not chunk:
                break
            yield chunk

def process_queries_chunk(args):
        n_rows, queries_chunk, n_queries = args
        chunk_size = len(queries_chunk)
        weight_chunk = np.zeros(n_rows)
        for j in range(chunk_size):
            answer_set = queries_chunk[j]
            query_weight = 1/n_queries
            per_element_weight = query_weight / len(answer_set)
            np.add.at(weight_chunk, answer_set, per_element_weight)
        return weight_chunk

def compute_weight_parallel(n_rows, queries, n_queries):
    """
        Parallel weight computation
    """
    weight = np.zeros(n_rows)
    n_cpus = cpu_count()
    chunk_size = (n_queries + n_cpus - 1) // n_cpus
    queries_g = chunked_iterable(queries.values(), chunk_size)

    with Pool(n_cpus) as pool:
        results = pool.map(process_queries_chunk, [(n_rows, chunk, n_queries) for chunk in queries_g])
    for weight_chunk in results:
        weight += weight_chunk
    return weight

def indep_df_parallel(n_rows, budget, queries, n_queries):
    """
        IndepDF algorithm: Weight computations computed in parallel
    """
    values = compute_weight_parallel(n_rows, queries, n_queries)
    top_indices = np.argpartition(values, -budget)[-budget:]
    return top_indices.tolist()


