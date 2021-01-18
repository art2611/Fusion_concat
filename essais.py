import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random

a = torch.randint(10, (2, 2))
b =  torch.randint(10, (2, 2))
print(a)
print(b)
print(torch.cat((a,b), 1))

sys.exit()
a = [[2,0], [4,6],[0,1]]
a = np.array(a)
print(a)

dist = pdist(a, metric='euclidean')
print(dist)
print(a.max(1))