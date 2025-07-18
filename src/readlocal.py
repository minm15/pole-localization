import numpy as np

# 1) 先 load npz，注意要 allow_pickle=True
data = np.load('/home/kaiii/pole-localization/nclt/2013-01-10/localmaps_6_0.25_0.08.npz', allow_pickle=True)
maps = data['maps']   # 这是一个 ndarray, dtype=object

# 2) 遍历每个 map，打印 shapes
for i, m in enumerate(maps):
    poleparams = m['poleparams']
    print(f"frame {i}: poleparams shape = {poleparams.shape}", end='')
    if 'descriptors' in m:
        desc = m['descriptors']
        print(f", descriptors shape = {desc.shape}")
    else:
        print(", no descriptors")