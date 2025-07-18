import numpy as np
import scipy
import util

class particlefilter:
    def __init__(self, count, start, posrange, angrange, 
            polemeans, polevar, descmap, descxy, T_w_o=np.identity(4), d_max = 1.0):
        self.p_min = 0.01
        self.d_max = d_max
        self.minneff = 0.5
        self.estimatetype = 'best'
        self.count = count
        r = np.random.uniform(low=0.0, high=posrange, size=[self.count, 1])
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=[self.count, 1])
        xy = r * np.hstack([np.cos(angle), np.sin(angle)])
        dxyp = np.hstack([xy, np.random.uniform(
            low=-angrange, high=angrange, size=[self.count, 1])])
        self.particles = np.matmul(start, util.xyp2ht(dxyp))
        self.weights = np.full(self.count, 1.0 / self.count)
        self.polemeans = polemeans
        self.descmap = descmap
        self.descxy = descxy
        self.poledist = scipy.stats.norm(loc=0.0, scale=np.sqrt(polevar))
        self.kdtree = scipy.spatial.cKDTree(polemeans[:, :2], leafsize=3)
        self.T_w_o = T_w_o
        self.T_o_w = util.invert_ht(self.T_w_o)

    @property
    def neff(self):
        return 1.0 / (np.sum(self.weights**2.0) * self.count)

    def update_motion(self, mean, cov):
        T_r0_r1 = util.xyp2ht(
            np.random.multivariate_normal(mean, cov, self.count))
        self.particles = np.matmul(self.particles, T_r0_r1)

    def update_measurement(self, desc, poleparams, resample=True):
        # matches = self.matcher(desc)
        # M = poleparams.shape[0]
        # polepos_r = np.hstack([poleparams[:, :2], np.zeros([M, 1]), np.ones([M, 1])]).T
        # print(polepos_r)
        # for pi in range(self.count):
        #     w = 1.0
        #     T = self.particles[pi] # 4×4
        #     # print(T)
        #     for i_local, j_global in matches:
        #         # global pole 在 m→w→r 投影：descpos 是 w 座標下 xy
        #         polepos_w = T.dot(polepos_r)
        #         print(desc[i_local])
        #         print(self.descmap[j_global])
        #         print(polepos_w[i_local][:2])
        #         print(self.descxy[j_global][:2])
        #         d = np.linalg.norm(polepos_w[i_local][:2] - self.descxy[j_global][:2])
        #         # print(d)
        #         w *= (self.poledist.pdf(min(d, self.d_max)) + 0.2)
        #     self.weights[pi] *= w
        # self.weights /= np.sum(self.weights)
        # if resample and self.neff < self.minneff:
        #     self.resample()
        
        n = poleparams.shape[0]
        polepos_r = np.hstack(
            [poleparams[:, :2], np.zeros([n, 1]), np.ones([n, 1])]).T
        for i in range(self.count):
            polepos_w = self.particles[i].dot(polepos_r)
            d, _ = self.kdtree.query(
                polepos_w[:2].T, k=1, distance_upper_bound=self.d_max)
            self.weights[i] *= np.prod(
                self.poledist.pdf(np.clip(d, 0.0, self.d_max)) + 0.2)
        self.weights /= np.sum(self.weights)

        if resample and self.neff < self.minneff:
            self.resample()

    def estimate_pose(self):
        if self.estimatetype == 'mean':
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights),
                    util.average_angles(xyp[:, 2], weights=self.weights)])
            return self.T_w_o.dot(util.xyp2ht(mean))
        if self.estimatetype == 'max':
            return self.particles[np.argmax(self.weights)]
        if self.estimatetype == 'best':
            i = np.argsort(self.weights)[-int(0.1 * self.count):]
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles[i]))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights[i]),
                    util.average_angles(xyp[:, 2], weights=self.weights[i])])                
            return self.T_w_o.dot(util.xyp2ht(mean))

    def resample(self):
        cumsum = np.cumsum(self.weights)
        pos = np.random.rand() / self.count
        idx = np.empty(self.count, dtype=np.int)
        ics = 0
        for i in range(self.count):
            while cumsum[ics] < pos:
                ics += 1
            idx[i] = ics
            pos += 1.0 / self.count
        self.particles = self.particles[idx]
        self.weights[:] = 1.0 / self.count
        
    def matcher(self, local_descs):
        """
        For each local descriptor row d1, find the index j of the global descriptor d2
        that maximizes the count of matching nonzero slots within tolerance 0.2.
        Returns a list of (i_local, j_global).
        """
        matches = []
        nz_local  = local_descs != 0
        nz_global = self.descmap != 0

        for i, d1 in enumerate(local_descs):
            best_j, best_score = -1, -1
            mask1 = nz_local[i]
            # compare against every global descriptor
            for j, d2 in enumerate(self.descmap):
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