import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.interpolate
import scipy.special
import cluster
import particlefilter
import pynclt
import util
import poles_extractor

mapextent = np.array([30.0, 30.0, 5.0])
mapsize = np.full(3, 0.2)
mapshape = np.array(mapextent / mapsize, dtype=np.int)
mapinterval = 0.25
mapdistance = 0.25
remapdistance = 5.0
n_mapdetections = 6
n_locdetections = 2
n_localmaps = 6

T_mc_r = pynclt.T_w_o
T_r_mc = util.invert_ht(T_mc_r)
T_m_mc = np.identity(4)
T_m_mc[:3, 3] = np.hstack([0.5 * mapextent[:2], 0.5])
T_mc_m = util.invert_ht(T_m_mc)
T_m_r = T_m_mc.dot(T_mc_r)
T_r_m = util.invert_ht(T_m_r)

def get_globalmapname():
    return 'globalmap_{:.0f}_{:.2f}_{:.2f}'.format(
        n_mapdetections, mapinterval, 0.08)


def get_locfileprefix():
    return 'localization_{:.0f}_{:.2f}_{:.2f}_{:.2f}'.format(
        n_mapdetections, mapinterval, 0.08, 0.20)


def get_localmapfile():
    return 'localmaps_{:.0f}_{:.2f}_{:.2f}.npz'.format(
        n_mapdetections, mapinterval, 0.08)


def get_evalfile():
    return 'evaluation_{:.0f}_{:.2f}_{:.2f}.npz'.format(
        n_mapdetections, mapinterval, 0.08)


def get_map_indices(session):
    distance = np.hstack([0.0, np.cumsum(np.linalg.norm(
        np.diff(session.T_w_r_gt_velo[:, :3, 3], axis=0), axis=1))])
    istart = []
    imid = []
    iend = []
    i = 0
    j = 0
    k = 0
    for id, d in enumerate(distance):
        if d >= i * mapinterval:
            istart.append(id)
            i += 1
        if d >= j * mapinterval + 0.5 * mapdistance:
            imid.append(id)
            j += 1
        if d > k * mapinterval + mapdistance:
            iend.append(id)
            k += 1
    return istart[:len(iend)], imid[:len(iend)], iend


def save_global_map(use_desc=False):
    globalmappos = np.empty([0, 2])
    mapfactors = np.full(len(pynclt.sessions), np.nan)
    poleparams = np.empty([0, 3])
    all_descs = []
    for isession, s in enumerate(pynclt.sessions):
        print(s)
        session = pynclt.session(s)
        istart, imid, iend = get_map_indices(session)
        localmappos = session.T_w_r_gt_velo[imid, :2, 3]
        if globalmappos.size == 0:
            imaps = range(localmappos.shape[0])
        else:
            imaps = []
            for imap in range(localmappos.shape[0]):
                distance = np.linalg.norm(
                    localmappos[imap] - globalmappos, axis=1).min()
                if distance > remapdistance:
                    imaps.append(imap)
        globalmappos = np.vstack([globalmappos, localmappos[imaps]])
        mapfactors[isession] = np.true_divide(len(imaps), len(imid))

        with progressbar.ProgressBar(max_value=len(imaps)) as bar:
            for iimap, imap in enumerate(imaps):
                iscan = imid[imap]
                xyz, _ = session.get_velo(iscan)
                
                if use_desc:
                    localpoleparam, desc = poles_extractor.detect_poles(
                        xyz, desc=True)
                    # if len(desc) < 4: continue 
                    all_descs.append(desc)
                else:
                    localpoleparam = poles_extractor.detect_poles(xyz)

                # localpoleparam = poles_extractor.detect_poles(xyz)
                localpoleparam_xy = localpoleparam[:, :2]
                localpoleparam_xy = localpoleparam_xy.T
                localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
                localpoleparam_xy = np.matmul(session.T_w_r_gt_velo[imid[imap]], localpoleparam_xy)
                localpoleparam[:, :2] = localpoleparam_xy[:2,:].T
                poleparams = np.vstack([poleparams, localpoleparam])

                bar.update(iimap)

    xy = poleparams[:, :2]
    a = poleparams[:, [2]]
    boxes = np.hstack([xy - a, xy + a])
    clustermeans = np.empty([0, 3])
    clusterdescs = np.empty((0,110))
    descs_array = np.vstack(all_descs)
    print(poleparams.shape, descs_array.shape)
    
    desc_txt_path = os.path.join(session.dir, 'cluster_descriptors.txt')
    cid = 0
    with open(desc_txt_path, 'w') as f_desc:
    
        for ci in cluster.cluster_boxes(boxes):
            ci = list(ci)
            if len(ci) < n_mapdetections:
                continue
            
            # build my mean pole and descriptor
            local_poleparams = poleparams[ci, :]      # (Nc, 3)
            local_descs = descs_array[ci, :]
            
            cluster_means, cluster_descs = merge_cluster_descriptors(
                local_poleparams,
                local_descs,
                threshold=0.2
            )
            clustermeans = np.vstack([clustermeans, cluster_means])
            clusterdescs = np.vstack([clusterdescs, cluster_descs])
            
            # clustermeans = np.vstack([clustermeans, np.average(poleparams[ci, :], axis=0)])
            # cluster_descs = np.vstack([cluster_descs, np.mean(descs_array[ci, :], axis=0)])
            
            f_desc.write(f"c{cid}\n")
            for i in range(len(cluster_descs)):
                vec = cluster_descs[i]
                line = " ".join(f"{v:.2f}" for v in vec)
                f_desc.write(line + "\n")
            # for pole_idx in ci:
            #     vec = descs_array[pole_idx]        # shape (180,)
            #     line = " ".join(f"{v:.2f}" for v in vec)
            #     f_desc.write(line + "\n")
            cid += 1
        
    globalmapfile = os.path.join('nclt', get_globalmapname() + '.npz')
    allglobalmapfile = os.path.join('nclt', get_globalmapname() + '_all' + '.npz')
    if use_desc:
        # concat all_descs (list of (Ni,180)) 成 (sum Ni,180)
        all_descs = np.vstack(all_descs)
        np.savez(globalmapfile,
                 polemeans=clustermeans,
                 descmeans=clusterdescs,
                 mapfactors=mapfactors,
                 mappos=globalmappos,
                 allpole=poleparams,
                 descriptors=descs_array)
        
        # np.savez(allglobalmapfile, polemeans=xy, descriptors=descs_array)
        print(f"allpole.shape: {poleparams.shape}, descriptors.shape: {descs_array.shape}")
        print(f"polemeans.shape: {clustermeans.shape}, descmeans.shape: {clusterdescs.shape}")
        
    else:
        np.savez(globalmapfile,
                 polemeans=clustermeans,
                 mapfactors=mapfactors,
                 mappos=globalmappos)
    plot_global_map(globalmapfile)


def plot_global_map(globalmapfile):
    data = np.load(globalmapfile)
    x, y = data['polemeans'][:, :2].T
    plt.clf()
    plt.scatter(x, y, s=1, c='b', marker='.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(globalmapfile[:-4] + '.svg')
    plt.savefig(globalmapfile[:-4] + '.pgf')
    print(data['mapfactors'])


def save_local_maps(sessionname, visualize=False, use_desc=False):
    print(sessionname)
    session = pynclt.session(sessionname)
    util.makedirs(session.dir)
    istart, imid, iend = get_map_indices(session)
    maps = []
    all_descs = []
    with progressbar.ProgressBar(max_value=len(iend)) as bar:
        for i in range(len(iend)):
            T_w_mc = util.project_xy(
                session.T_w_r_odo_velo[imid[i]].dot(T_r_mc))
            T_w_m = T_w_mc.dot(T_mc_m)
            T_m_w = util.invert_ht(T_w_m)
            T_w_r = session.T_w_r_odo_velo[imid[i]]
            T_m_r = np.matmul(T_m_w, T_w_r)
            T_g_mc = util.project_xy(session.T_w_r_gt_velo[imid[i]].dot(T_r_mc))
            T_g_m = T_g_mc.dot(T_mc_m)

            iscan = imid[i]
            xyz, _ = session.get_velo(iscan)
            
            if use_desc:
                poleparams, desc = poles_extractor.detect_poles(
                    xyz, desc=True)
                all_descs.append(desc)
            else:
                poleparams = poles_extractor.detect_poles(xyz)

            # poleparams = poles_extractor.detect_poles(xyz)
            localpoleparam_xy = poleparams[:, :2]
            localpoleparam_xy = localpoleparam_xy.T
            localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
            localpoleparam_xy = np.matmul(T_m_r, localpoleparam_xy)
            poleparams[:, :2] = localpoleparam_xy[:2,:].T

            map = {'poleparams': poleparams, 'T_w_m': T_w_m, 'T_g_m': T_g_m,
                'istart': istart[i], 'imid': imid[i], 'iend': iend[i]}
            if use_desc:
                map['descriptors'] = desc   # (Ni, 180) array
            maps.append(map)
            bar.update(i)
            
    np.savez(os.path.join(session.dir, get_localmapfile()), maps=maps)

def localize(sessionname, visualize=False, use_desc=False):
    print(sessionname)
    
    # load the global map data
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans'][:, :2]
    descmap = mapdata['descmeans']
    descxy = mapdata['polemeans'][:, :2]
    # descmap = mapdata['descriptors']
    # descxy = mapdata['allpole'][:, :2]
    print(f"descmap size: {descmap.shape}, descxy size: {descxy.shape}")
    polevar = 1.50
    
    # load the local map data
    session = pynclt.session(sessionname)
    locdata = np.load(os.path.join(session.dir, get_localmapfile()), allow_pickle=True)['maps']
    polepos_m = []
    polepos_w = []
    desclocal = []
    for i in range(len(locdata)):
        n = locdata[i]['poleparams'].shape[0]
        pad = np.hstack([np.zeros([n, 1]), np.ones([n, 1])])
        polepos_m.append(np.hstack([locdata[i]['poleparams'][:, :2], pad]).T)
        polepos_w.append(locdata[i]['T_w_m'].dot(polepos_m[i]))
        desclocal.append(locdata[i]['descriptors'])
        
    istart = 0
    T_w_r_start = util.project_xy(
        session.get_T_w_r_gt(session.t_relodo[istart]).dot(T_r_mc)).dot(T_mc_r)
    
    # construct the descmap index
    descmap_index, edges = build_descmap_index(descmap, 32)
    
    filter = particlefilter.particlefilter(500, 
        T_w_r_start, 2.5, np.radians(5.0), polemap, polevar, descmap, descxy, descmap_index, edges, T_w_o=T_mc_r)
    filter.estimatetype = 'best'
    filter.minneff = 0.5

    if visualize:
        plt.ion()
        figure = plt.figure()
        nplots = 1
        mapaxes = figure.add_subplot(nplots, 1, 1)
        mapaxes.set_aspect('equal')
        mapaxes.scatter(polemap[:, 0], polemap[:, 1], s=5, c='b', marker='s')
        x_gt, y_gt = session.T_w_r_gt[::20, :2, 3].T
        mapaxes.plot(x_gt, y_gt, 'g')
        particles = mapaxes.scatter([], [], s=1, c='r')
        arrow = mapaxes.arrow(0.0, 0.0, 1.0, 0.0, length_includes_head=True, 
            head_width=0.7, head_length=1.0, color='k')
        arrowdata = np.hstack(
            [arrow.get_xy(), np.zeros([8, 1]), np.ones([8, 1])]).T
        locpoles = mapaxes.scatter([], [], s=30, c='k', marker='x')
        viewoffset = 25.0

    imap = 0
    while imap < locdata.shape[0] - 1 and \
            session.t_velo[locdata[imap]['iend']] < session.t_relodo[istart]:
        imap += 1
    T_w_r_est = np.full([session.t_relodo.size, 4, 4], np.nan)
    with progressbar.ProgressBar(max_value=session.t_relodo.size) as bar:
        for i in range(istart, session.t_relodo.size):
            relodocov = np.empty([3, 3])
            relodocov[:2, :2] = session.relodocov[i, :2, :2]
            relodocov[:, 2] = session.relodocov[i, [0, 1, 5], 5]
            relodocov[2, :] = session.relodocov[i, 5, [0, 1, 5]]
            filter.update_motion(session.relodo[i], relodocov * 2.0**2)
            T_w_r_est[i] = filter.estimate_pose()
            t_now = session.t_relodo[i]
            if imap < locdata.shape[0]:
                t_end = session.t_velo[locdata[imap]['iend']]
                if t_now >= t_end:
                    imaps = range(imap, np.clip(imap-n_localmaps, -1, None), -1)
                    xy = np.hstack([polepos_w[j][:2] for j in imaps]).T
                    a = np.vstack([ld['poleparams'][:, [2]] \
                        for ld in locdata[imaps]])
                    boxes = np.hstack([xy - a, xy + a])
                    ipoles = set(range(polepos_w[imap].shape[1]))
                    iactive = set()
                    for ci in cluster.cluster_boxes(boxes):
                        if len(ci) >= n_locdetections:
                            iactive |= set(ipoles) & ci
                    iactive = list(iactive)
                    if len(iactive) >= 4:
                        t_mid = session.t_velo[locdata[imap]['imid']]
                        T_w_r_mid = util.project_xy(session.get_T_w_r_odo(
                            t_mid).dot(T_r_mc)).dot(T_mc_r)
                        T_w_r_now = util.project_xy(session.get_T_w_r_odo(
                            t_now).dot(T_r_mc)).dot(T_mc_r)
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(T_r_m).dot(
                            polepos_m[imap][:, iactive])
                        filter.update_measurement(desclocal[imap][iactive], polepos_r_now[:2].T)
                        T_w_r_est[i] = filter.estimate_pose()
                        if visualize:
                            polepos_w_est = T_w_r_est[i].dot(polepos_r_now)
                            locpoles.set_offsets(polepos_w_est[:2].T)

                    imap += 1
            
            if visualize:
                particles.set_offsets(filter.particles[:, :2, 3])
                arrow.set_xy(T_w_r_est[i].dot(arrowdata)[:2].T)
                x, y = T_w_r_est[i, :2, 3]
                mapaxes.set_xlim(left=x - viewoffset, right=x + viewoffset)
                mapaxes.set_ylim(bottom=y - viewoffset, top=y + viewoffset)
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            bar.update(i)
    filename = os.path.join(session.dir, get_locfileprefix() \
        + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.npz'))
    np.savez(filename, T_w_r_est=T_w_r_est)

def plot_trajectories():
    trajectorydir = os.path.join(
        pynclt.resultdir, 'trajectories_est_{:.0f}_{:.0f}_{:.2f}'.format(
            n_mapdetections, n_locdetections, mapinterval))
    pgfdir = os.path.join(trajectorydir, 'pgf')
    util.makedirs(trajectorydir)
    util.makedirs(pgfdir)
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans']
    for sessionname in pynclt.sessions:
        try:
            session = pynclt.session(sessionname)
            files = [file for file \
                in os.listdir(os.path.join(pynclt.resultdir, sessionname)) \
                    if file.startswith(get_locfileprefix())]
            for file in files:
                T_w_r_est = np.load(os.path.join(
                    pynclt.resultdir, sessionname, file))['T_w_r_est']
                plt.clf()
                plt.scatter(polemap[:, 0], polemap[:, 1], 
                    s=1, c='b', marker='.')
                plt.plot(session.T_w_r_gt[::20, 0, 3], 
                    session.T_w_r_gt[::20, 1, 3], color=(0.5, 0.5, 0.5))
                plt.plot(T_w_r_est[::20, 0, 3], T_w_r_est[::20, 1, 3], 'r')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.gcf().subplots_adjust(
                    bottom=0.13, top=0.98, left=0.145, right=0.98)
                filename = sessionname + file[18:-4]
                plt.savefig(os.path.join(trajectorydir, filename + '.svg'))
                plt.savefig(os.path.join(pgfdir, filename + '.pgf'))
        except:
            pass


def evaluate():
    stats = []
    for sessionname in pynclt.sessions:
        files = [file for file \
            in os.listdir(os.path.join(pynclt.resultdir, sessionname)) \
                if file.startswith(get_locfileprefix())]
        files.sort()
        session = pynclt.session(sessionname)
        cumdist = np.hstack([0.0, np.cumsum(np.linalg.norm(np.diff(
            session.T_w_r_gt[:, :3, 3], axis=0), axis=1))])
        t_eval = scipy.interpolate.interp1d(
            cumdist, session.t_gt)(np.arange(0.0, cumdist[-1], 1.0))
        T_w_r_gt = np.stack([util.project_xy(
                session.get_T_w_r_gt(t).dot(T_r_mc)).dot(T_mc_r) \
                    for t in t_eval])
        T_gt_est = []
        for file in files:
            T_w_r_est = np.load(os.path.join(
                pynclt.resultdir, sessionname, file))['T_w_r_est']
            T_w_r_est_interp = np.empty([len(t_eval), 4, 4])
            iodo = 1
            inum = 0
            for ieval in range(len(t_eval)):
                while session.t_relodo[iodo] < t_eval[ieval]:
                    iodo += 1
                    if iodo >= session.t_relodo.shape[0]:
                        break
                if iodo >= session.t_relodo.shape[0]:
                        break
                T_w_r_est_interp[ieval] = util.interpolate_ht(
                    T_w_r_est[iodo-1:iodo+1], 
                    session.t_relodo[iodo-1:iodo+1], t_eval[ieval])
                inum += 1
            T_gt_est.append(
                np.matmul(util.invert_ht(T_w_r_gt), T_w_r_est_interp)[:inum,...])
        T_gt_est = np.stack(T_gt_est)
        lonerror = np.mean(np.mean(np.abs(T_gt_est[..., 0, 3]), axis=-1))
        laterror = np.mean(np.mean(np.abs(T_gt_est[..., 1, 3]), axis=-1))
        poserrors = np.linalg.norm(T_gt_est[..., :2, 3], axis=-1)
        poserror = np.mean(np.mean(poserrors, axis=-1))
        posrmse = np.mean(np.sqrt(np.mean(poserrors**2, axis=-1)))
        angerrors = np.degrees(np.abs(
            np.array([util.ht2xyp(T)[:, 2] for T in T_gt_est])))
        angerror = np.mean(np.mean(angerrors, axis=-1))
        angrmse = np.mean(np.sqrt(np.mean(angerrors**2, axis=-1)))
        stats.append({'session': sessionname, 'lonerror': lonerror, 
            'laterror': laterror, 'poserror': poserror, 'posrmse': posrmse,
            'angerror': angerror, 'angrmse': angrmse, 'T_gt_est': T_gt_est})
        
        # save figure
        pos_errors_time = np.linalg.norm(T_gt_est[..., :2, 3], axis=-1)
        mean_pos_errors = pos_errors_time.mean(axis=0)
        plt.figure()
        plt.plot(t_eval, mean_pos_errors, linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Mean Position Error [m]')
        plt.title(f'Session {sessionname} Position Error')
        outdir = os.path.join(pynclt.resultdir, sessionname)
        img_path = os.path.join(outdir, 'poserror_over_time.png')
        plt.savefig(img_path, dpi=150)
        plt.close()
        
    np.savez(os.path.join(pynclt.resultdir, get_evalfile()), stats=stats)
    
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    print('session \t f\te_pos \trmse_pos \te_ang \te_rmse')
    row = '{session} \t{f} \t{poserror} \t{posrmse} \t{angerror} \t{angrmse}'
    for i, stat in enumerate(stats):
        print(row.format(
            session=stat['session'],
            f=mapdata['mapfactors'][i] * 100.0,
            poserror=stat['poserror'],
            posrmse=stat['posrmse'],
            angerror=stat['angerror'],
            angrmse=stat['angrmse']))
        
def merge_cluster_descriptors(poleparams, descs_array, threshold=0.2):
    """
    input:
      poleparams:  np.ndarray of shape (N, 3)
      descs_array: np.ndarray of shape (N, D)
      threshold:   float, 
    output:
      cluster_means: list of np.ndarray (shape (3,))  M x 3
      cluster_descs: list of np.ndarray (shape (D,))  M x D
    """
    N, D = descs_array.shape

    # mask zero-only descriptors
    zero_mask = np.all(descs_array == 0, axis=1)
    zero_idxs = np.where(zero_mask)[0].tolist()
    nonzero_idxs = np.where(~zero_mask)[0].tolist()

    # sort by dim-0
    nonzero_idxs.sort(key=lambda i: descs_array[i, 0])

    clusters = []
    cluster_means = [] 
    cluster_mins  = []
    cluster_maxs  = []

    # cluster by order
    for idx in nonzero_idxs:
        desc_new = descs_array[idx]
        if not clusters:
            # the first cluster
            clusters.append([idx])
            cluster_means.append(desc_new.copy())
            cluster_mins.append(desc_new.copy())
            cluster_maxs.append(desc_new.copy())
        else:
            last = len(clusters) - 1
            mean = cluster_means[last]
            mins = cluster_mins[last]
            maxs = cluster_maxs[last]
            M = len(clusters[last])

            # try to combine
            can_merge = True
            for d in range(D):
                v0 = mean[d]
                v1 = desc_new[d]
                if v0 != 0 and v1 != 0:
                    new_mean = (v0 * M + v1) / (M + 1)
                    new_min  = min(mins[d],    v1)
                    new_max  = max(maxs[d],    v1)
                    if abs(new_mean - new_min) > threshold or \
                       abs(new_max - new_mean) > threshold:
                        can_merge = False
                        break

            if can_merge:
                # update the stat of new combined cluster
                clusters[last].append(idx)
                for d in range(D):
                    v0 = mean[d]
                    v1 = desc_new[d]
                    if v0 == 0 and v1 != 0:
                        mean[d] = v1
                    elif v0 != 0 and v1 != 0:
                        mean[d] = (v0 * M + v1) / (M + 1)
                        
                    if v1 != 0:
                        mins[d] = min(mins[d], v1)
                        maxs[d] = max(maxs[d], v1)
                
                cluster_means[last] = mean
                cluster_mins[last]  = mins
                cluster_maxs[last]  = maxs
            else:
                # create a new cluster
                clusters.append([idx])
                cluster_means.append(desc_new.copy())
                cluster_mins.append(desc_new.copy())
                cluster_maxs.append(desc_new.copy())

    # calculate the final cluster means and descs
    final_pole_means = []
    final_desc_means = []
    for k, idxs in enumerate(clusters):
        # pole means: include the pole for zero-only desc
        all_idxs = idxs + zero_idxs
        pm = np.mean(poleparams[all_idxs, :], axis=0)
        final_pole_means.append(pm)
        # desc means: cluster_means[k]
        final_desc_means.append(cluster_means[k].copy())

    # output np.ndarray
    if len(final_pole_means) == 0:
        return np.zeros((0, 3)), np.zeros((0, D))
    
    clustermeans  = np.vstack(final_pole_means)   # shape (M, 3)
    cluster_descs = np.vstack(final_desc_means)   # shape (M, D)
    return clustermeans, cluster_descs

def build_descmap_index(descmap, k):
    """
    divide descmap[:,0] to k parts, return:
      - descmap_index: dict[int, List[int]], bucket number -> global descriptor index list
      - edges: np.ndarray of shape (k+1,), edge number for dividing bucket

    Usage:
      descmap_index, edges = build_descmap_index(descmap, k=16)
    """
    dim0 = descmap[:, 0]
    edges = np.quantile(dim0, np.linspace(0, 1, k+1))
    
    descmap_index = {i: [] for i in range(k)}
    
    for j, v in enumerate(dim0):
        bin_idx = np.searchsorted(edges, v, side='right') - 1
        bin_idx = max(0, min(bin_idx, k-1))
        descmap_index[bin_idx].append(j)
    
    return descmap_index, edges


if __name__ == '__main__':
    #save_global_map(use_desc=True)
    for session in pynclt.sessions:
        #save_local_maps(session, use_desc=True)
        localize(session, visualize=False, use_desc=False)

    plot_trajectories()
    evaluate()
    
    
# for session in sessions:
#     for all mappos do clustermeans, clusterdescs calculation
#     if this is the first session:
#         add all of the clusterdescs and clustermeans into the global map
#     else:
#         for each desc in cluster descs:
#             do_match(desc, globaldescs)
#             if the matched pair <i, j> satisfy distance(clustermeans[i], globalpolexy[j]) < 1.0:
#                 continue
#             else:
#                 add the desc and corresponding pole clustermean into the global map
