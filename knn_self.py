import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import warnings
from matplotlib import style
from collections import Counter

style.use("fivethirtyeight")

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_feature = [3,4]

def k_nearest_neighbor(data, predict,k=3):
    if len(data)>=k:
        warnings.warn('chutiya hai kya?')
    distances = []
    for group in data:
        for feature in data[group]:
            euclidean_dist= np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([euclidean_dist,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common())
    vote_result = Counter(votes).most_common()[0][0]
    return vote_result

prediction = k_nearest_neighbor(dataset, new_feature,k=3)
print('The new feature belongs to class/cluster:', prediction)