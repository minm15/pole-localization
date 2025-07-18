import os
import numpy as np
import matplotlib.pyplot as plt

# === 请根据实际路径修改 ===
GLOBAL_MAP_NPZ = 'nclt/globalmap_6_0.25_0.08_all.npz'
LOCAL_MAPS_NPZ = 'nclt/2013-01-10/localmaps_6_0.25_0.08.npz'

OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 载入全局 poles（世界坐标系）
gdata = np.load(GLOBAL_MAP_NPZ)
global_poles = gdata['polemeans'][:, :2]  # shape (M,2)

# 2. 载入所有帧的 local poles 及其变换矩阵
ldata = np.load(LOCAL_MAPS_NPZ, allow_pickle=True)
local_frames = ldata['maps']  # list of dict

# 3. 累积所有帧的 world 坐标下的 local poles
all_local_pts = []
for frame in local_frames:
    poleparams = frame['poleparams']   # (Ni,3)
    T_w_m      = frame['T_w_m']        # (4,4)
    T_g_m = frame['T_g_m']
    Ni = poleparams.shape[0]
    if Ni == 0:
        continue
    # 构造齐次坐标 (Ni,4)
    pts_m = np.hstack([
        poleparams[:, :2],
        np.zeros((Ni,1)),
        np.ones((Ni,1))
    ])
    # 变换到世界坐标系 (Ni,2)
    pts_w = (T_g_m.dot(pts_m.T))[:2].T
    all_local_pts.append(pts_w)

if all_local_pts:
    all_local_pts = np.vstack(all_local_pts)
else:
    all_local_pts = np.empty((0,2))

# 4. 绘图
plt.figure(figsize=(10,10))
plt.scatter(global_poles[:,0], global_poles[:,1],
            c='red',   s=5, label='Global poles')
plt.scatter(all_local_pts[:,0], all_local_pts[:,1],
            c='blue',  s=2, label='All local poles')
plt.axis('equal')
plt.legend(loc='upper right')
plt.title('Global vs. All Local Poles')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.tight_layout()

# 5. 保存
out_path = os.path.join(OUTPUT_DIR, 'all_local_vs_global.png')
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved combined visualization to {out_path}")