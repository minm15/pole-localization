import numpy as np

# 载入 .npz
data = np.load('nclt/globalmap_6_0.25_0.08_all.npz')

# 列出所有键
print(data.files)
# >>> ['polemeans', 'mapfactors', 'mappos', 'descriptors']

# 访问各个数组
polemeans   = data['polemeans']    # shape: (M, 3)   每行 [x, y, radius]
descriptors = data['descriptors']  # shape: (N, 180) 所有 pole 的 180D 描述子

# 例如：
print("polemeans shape   :", polemeans.shape)
print("descriptors shape :", descriptors.shape)