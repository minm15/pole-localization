#!/usr/bin/env python3
# print_local_desc.py

import numpy as np
import os

# 請替換為你的 localmaps 檔案路徑
LOCALMAP_FILE = '/home/kaiii/pole-localization/nclt/2013-01-10/localmaps_6_0.25_0.08.npz'

def main():
    # 讀取 npz，maps 裡面是 dict list
    data = np.load(LOCALMAP_FILE, allow_pickle=True)
    maps = data['maps']

    for frame_idx, m in enumerate(maps):
        desc = m.get('descriptors', None)
        if desc.shape[0] < 5: continue
        if desc is None:
            print(f"[Frame {frame_idx}] 沒有 descriptors，poleparams shape = {m['poleparams'].shape}")
        else:
            print(f"[Frame {frame_idx}] descriptors shape = {desc.shape}")
            # 如果想看前兩行前十維度：
            for i in range(desc.shape[0]):
                vals = ', '.join(f"{v:.3f}" for v in desc[i])
                print(f"  desc[{i}][0:10] = [{vals}, ...]")
    print("Done.")

if __name__ == '__main__':
    main()