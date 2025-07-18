#!/usr/bin/env python
# src/test_desc_match.py

import numpy as np
import progressbar
from scipy.spatial import cKDTree

# === paths: adjust as needed ===
GLOBAL_MAP_NPZ = '/home/kaiii/pole-localization/nclt/globalmap_6_0.25_0.08_all.npz'
LOCAL_MAPS_NPZ = '/home/kaiii/pole-localization/nclt/2013-01-10/localmaps_6_0.25_0.08.npz'

def matcher(local_descs, global_descs):
    """
    For each local descriptor row d1, find the index j of the global descriptor d2
    that maximizes the count of matching nonzero slots within tolerance 0.2.
    Returns a list of (i_local, j_global).
    """
    matches = []
    nz_local  = local_descs != 0
    nz_global = global_descs != 0

    for i, d1 in enumerate(local_descs):
        best_j, best_score = -1, -1
        mask1 = nz_local[i]
        # compare against every global descriptor
        for j, d2 in enumerate(global_descs):
            common = mask1 & nz_global[j]
            if not np.any(common):
                score = 0
            else:
                diffs = np.abs(d1[common] - d2[common])
                score = int((diffs <= 0.2).sum())
            if score > best_score:
                best_score, best_j = score, j
        matches.append((i, best_j))
    return matches

def compute_geometric_distances(local_params, global_params, matches):
    """
    Given matches list of (i_local, j_global), compute Euclidean distance
    between the (x,y) of each matched pair.
    """
    dists = []
    for i_local, j_global in matches:
        p_loc = local_params[i_local, :2]
        p_glb = global_params[j_global, :2]
        dists.append(np.linalg.norm(p_loc - p_glb))
    return dists

def main():
    # load global map
    gdata = np.load(GLOBAL_MAP_NPZ)
    global_poles   = gdata['polemeans']    # (M,3)
    global_descs   = gdata['descriptors']  # (N,180)

    # load local maps
    ldata = np.load(LOCAL_MAPS_NPZ, allow_pickle=True)
    local_maps = ldata['maps']              # array of dicts

    total_poles       = 0
    attempted_matches = 0
    correct_matches   = 0

    bar = progressbar.ProgressBar(max_value=len(local_maps))
    for idx, m in enumerate(local_maps):
        desc = m.get('descriptors', None)
        params = m['poleparams']  # (Ni,3)
        
        params_local = m['poleparams']
        T_g_m = m['T_g_m']

        # skip small frames
        if desc is None or desc.shape[0] < 5:
            bar.update(idx)
            continue

        Ni = desc.shape[0]
        total_poles += Ni
        
        # transform to world frame
        pts = np.hstack([
            params_local[:, :2],
            np.zeros((Ni,1)),
            np.ones ((Ni,1))
        ])
        world_xy = (T_g_m.dot(pts.T))[:2].T

        # 1) match descriptors
        matches = matcher(desc, global_descs)
        attempted_matches += len(matches)

        # 2) compute geometry distances
        dists = compute_geometric_distances(world_xy, global_poles, matches)

        # 3) count correct (<1m)
        correct_matches += sum(1 for d in dists if d < 1.0)

        bar.update(idx)

    # final summary
    print("\n=== Summary ===")
    print("Total local poles            :", total_poles)
    print("Total attempted matches      :", attempted_matches)
    print("Total correct (<1m) matches  :", correct_matches)
    if attempted_matches > 0:
        print("Accuracy                     : {:.1f}%".format(
            100.0 * correct_matches / attempted_matches))

if __name__ == '__main__':
    main()