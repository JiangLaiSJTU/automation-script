import numpy as np

# 创建一个三维数组
array = np.array([[[1, 2, 3],
                   [4, 5, 6]],
                  [[7, 8, 9],
                   [10, 11, 12]]])

# 沿着第一个轴（深度）求和
sum_axis0 = np.sum(array, axis=0)
print("沿着第一个轴（深度）求和结果:")
print(sum_axis0)

# 沿着第二个轴（行）求和
sum_axis1 = np.sum(array, axis=1)
print("沿着第二个轴（行）求和结果:")
print(sum_axis1)

# 沿着第三个轴（列）求和
sum_axis2 = np.sum(array, axis=2)
print("沿着第三个轴（列）求和结果:")
print(sum_axis2)
