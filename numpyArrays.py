import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]], float)

print(arr[1, 1])
print(arr[1,:])  # первая строка (еще есть нулевая :) )
print(arr[:,0])  # нулевой столбец

