import numpy as np

# [1, 2, 3]
# [4, 5, 6]

arr = np.array([[1, 2, 3], [4, 5, 6]], float)

print(arr)
print(arr[1,:2])  # первая строка (еще есть нулевая :) )
print(arr[:,0])  # нулевой столбец

