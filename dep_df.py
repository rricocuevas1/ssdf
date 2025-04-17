import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def jaccard(d_1, d_2):
    """The Jaccard simmilarity"""
    k = len(d_1)
    intersection, union = 0, k
    for i in range(k):
        if d_1[i] == d_2[i]:
            intersection = intersection + 1
        else:  
            union = union + 1
    return intersection / union

def compute_similarity_matrix(data_array, answer_set_q, d, jaccard_sim):
    """Compute similarity vector"""
    if jaccard_sim:
        similarities = np.array([jaccard(d, data_array[i]) for i in answer_set_q])
    else:
        similarities = cosine_similarity([d], data_array[answer_set_q])[0]
    return similarities

def gradient(x, n, queries, p, m, data_array, K=1, jaccard_sim=True):
    """Double-stochastic estimation of the gradient of F"""
    if p==None:
        q_index = np.random.choice(m)
    else:
        q_index = np.random.choice(m, p=p)
    
    answer_set_q = queries[q_index]
    d_index = np.random.choice(answer_set_q)
    d = data_array[d_index]

    similarities = np.zeros(n)
    similarities[answer_set_q] = compute_similarity_matrix(data_array, answer_set_q, d, jaccard_sim)

    grad = np.zeros(n)
    for _ in range(K):
        D_prime = np.random.rand(n) < x
        answer_set_q_D_prime = np.intersect1d(np.where(D_prime)[0], answer_set_q)

        if answer_set_q_D_prime.size > 0:
            maximum = np.max(similarities[answer_set_q_D_prime])
        else:
            maximum = 0

        grad_contributions = np.maximum(0, similarities[answer_set_q] - maximum)
        grad[answer_set_q] += grad_contributions

    return grad / K

def largest_coordinates(d, n, B):
    """Select the B largest coordinates"""
    indices = np.argpartition(d, -B)[-B:]
    v = np.zeros(n, dtype=int)
    v[indices] = 1
    return v.tolist()

def scg_squared(n, B, queries, p, m, dataset, T=10000, K=1, jaccard_sim=True):
    """The SCG^2 algorithm"""
    data_array = dataset
    d = np.zeros(n)
    x = np.zeros(n)
    ro_base = 1 / 2

    for t in range(T):
        ro = ro_base / ((t+1) ** (2/3))
        grad = gradient(x, n, queries, p, m, data_array, K, jaccard_sim)
        d = d * (1 - ro) + grad * ro
        v = largest_coordinates(d, n, B)
        x += np.array(v) / T

    return x.tolist()

def pipage_rounding(x_fractional, n, B):
    """Randomized pipage rounding algorithm"""
    p = 0
    q = 1
    x = [round(i, 2) for i in x_fractional.copy()]
    if sum(x) != B:  # Normalize
        diff = B - sum(x)
        index = random.randrange(n)
        while x[index] == 0 or x[index] == 1:
            index = random.randrange(n)
        x[index] = x[index] + diff
    for t in range(n-1):
        if p >= n or q >= n:
            return x
        elif x[p] == 0 or x[p] == 1:
            p = max((p, q)) + 1
        elif x[q] == 0 or x[q] == 1:
            q = max((p, q)) + 1
        elif x[p] + x[q] < 1:
            # This means that \alpha_x = min{1-x[p], x[q]} = x[q]
            #                 \beta_x  = min{1-x[q], x[p]} = x[p]
            # Consequently probability = \alpha_x/(\alpha_x + \beta_x) = x[q] / (x[p] + x[q])
            if np.random.rand() < x[q]/(x[p] + x[q]):
                x[q] = x[p] + x[q]
                x[p] = 0
                p = max((p, q)) + 1
            else:
                x[p] = x[p] + x[q]
                x[q] = 0
                q = max((p, q)) + 1
        else:
            # This means that \alpha_x = min{1-x[p], x[q]} = 1- x[p]
            #                 \beta_x  = min{1-x[q], x[p]} = 1- x[q]
            # Consequently probability = \alpha_x/(\alpha_x + \beta_x) = 1- x[p] / (2 - x[p] - x[q])
            if np.random.rand() < (1 - x[p])/(2 - x[p] - x[q]):
                x[p] = x[p] + x[q] - 1
                x[q] = 1
                q = max((p, q)) + 1

            else:
                x[q] = x[p] + x[q] - 1
                x[p] = 1
                p = max((p, q)) + 1
    answer = []
    for i in range(n):
        if int(x[i]) == 1:
            answer.append(i)
    return answer
