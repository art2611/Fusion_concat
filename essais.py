import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

x = torch.randn(2, 3)
print(x)
cat = torch.cat((x,x), 1)
print(cat)
print(cat.shape)
print(x.shape)


sys.exit()
a = [[2,0], [4,6],[0,1]]
a = np.array(a)
print(a)

dist = pdist(a, metric='euclidean')
print(dist)
print(a.max(1))