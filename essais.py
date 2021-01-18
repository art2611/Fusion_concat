import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
a = [1,2,3]
b = [2,3]
a.extend(b)
print(a)
if 4 in [1,2]:
    print("true")
for k in range(0) :
    print("COUCOU")


sys.exit()
a = [[2,0], [4,6],[0,1]]
a = np.array(a)
print(a)

dist = pdist(a, metric='euclidean')
print(dist)
print(a.max(1))