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
