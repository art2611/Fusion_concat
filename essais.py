import numpy as np
import torch
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random


# list1 = [3,4,5]
# list2= []
# for k in range(2):
#     list2.extend(random.sample(list1, 4))
# print(list2)


# rgb_tensor = torch.tensor(np.array([[1, 2, 3], [3, 2, 1]]))
# ir_tensor = torch.tensor(np.array([[1, 2, 3], [1, 2, 3]]))
# summed_tensor = rgb_tensor.add(ir_tensor)
# print(f" summed_tensor : {summed_tensor}")
for k in range(4,4) :
    print(k)
sys.exit()

x = np.array([1, 3, 4, 6])
y = np.array([2, 3, 5, 1])
y2 = np.sin(x)
plt.plot(x, y, label="cos(x)")
plt.plot(x, y2, label="sin(x)")

plt.legend()
plt.xlabel("abscisses")
plt.ylabel("ordonnees")
plt.title("Fonction plop")

plt.show()

# affiche la figure a l'ecran
# a = torch.randint(10, (2, 2))
# b = torch.randint(10, (2, 2))
# print(a)
# print(b)
# c = torch.cat((a, b), dim=-1)
#
# print(c)
# print(c.shape)
sys.exit()


def fonction(x):
    return (x + 2)


a = {"bonjour": fonction, "salut": 3}
print(a["bonjour"](2))

a = [[2, 0], [4, 6], [0, 1]]
a = np.array(a)
print(a)

dist = pdist(a, metric='euclidean')
print(dist)
print(a.max(1))
