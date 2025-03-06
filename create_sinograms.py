import numpy as np

a = [1, 2, 3]
b = []
for i in range(len(a)):
    b.append(a[i])
a[2] = 5

print(b)
