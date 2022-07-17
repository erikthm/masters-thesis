import numpy as np
import multiprocessing as mp
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsela
from scipy import interpolate as interp

from EMequations import kirchoff_matrix, biot_savart, trace_mag_field


class GICi3D():
    def __init__(self, rad, theta, phi, date):

        rad_range = np.linspace(rad[0], rad[1], rad[2])
        theta_range = np.linspace(theta[0], theta[1], theta[2])
        phi_range = np.linspace(phi[0], phi[1], phi[2])

        self._radgrid, self._thetagrid, self._phigrid = np.meshgrid(rad_range, theta_range, phi_range, indexing='ij')

        if (rad[1] - rad[0])/rad[2] < 10:
            stepsize = (rad[1] - rad[0])/rad[2]
        else:
            stepsize = 10

        radgrid0 = self._radgrid[0, ...].flatten()
        thetagrid0 = self._thetagrid[0, ...].flatten()
        phigrid0 = self._phigrid[0, ...].flatten()
        rad_traced, theta_traced, phi_traced = trace_mag_field(radgrid0,
                                                               thetagrid0,
                                                               phigrid0,
                                                               radmax=rad[1],
                                                               stepsize=stepsize,
                                                               date=date)

        for ii in range(rad_traced.shape[1]):
            interp_theta = interp.interp1d(rad_traced[:, ii], theta_traced[:, ii])
            interp_phi = interp.interp1d(rad_traced[:, ii], phi_traced[:, ii])

            idx_theta, idx_phi = np.unravel_index(ii, self._radgrid.shape[1:])

            self._thetagrid[:, idx_theta, idx_phi] = interp_theta(rad_range)
            self._phigrid[:, idx_theta, idx_phi] = interp_phi(rad_range)

        self._current_models = {}

    @property
    def radgrid(self):
        return self._radgrid
    
    @property
    def thetagrid(self):
        return self._thetagrid

    @property
    def phigrid(self):
        return self._phigrid
    
    @property
    def shape(self):
        return self._radgrid.shape

    def get_current_model(self, name):
        return self._current_models[name]

    def add_current_model(self, currents, name):
        # TODO: Add check for the current model going outside grid limits
        
        current_model = np.zeros(self._radgrid.shape + (3,))

        for list in currents:
            key = list[0]
            values = list[1:]
            if key == 'FAC_in':
                r0, ampere = values
                idx0 = np.abs(self._thetagrid[0, ...] - r0[0]).argmin(axis=0)[0]
                idx1 = np.abs(self._phigrid[0, ...] - r0[1]).argmin(axis=1)[0]
                current_model[0, idx0, idx1, 0] = -1*ampere
                continue
            elif key == 'FAC_out':
                r0, ampere = values
                idx0 = np.abs(self._thetagrid[0, ...] - r0[0]).argmin(axis=0)[0]
                idx1 = np.abs(self._phigrid[0, ...] - r0[1]).argmin(axis=1)[0]
                current_model[0, idx0, idx1, 0] = ampere
                continue
            elif key == 'horizontal':
                r0, r1, ampere = values
                
            start_idx = np.array([np.abs(self._thetagrid[0, ...] - r0[0]).argmin(axis=0)[0],
                                  np.abs(self._phigrid[0, ...] - r0[1]).argmin(axis=1)[0]])
            stop_idx = np.array([np.abs(self._thetagrid[0, ...] - r1[0]).argmin(axis=0)[0],
                                 np.abs(self._phigrid[0, ...] - r1[1]).argmin(axis=1)[0]])

            local_mask = np.zeros(np.append((np.abs(stop_idx - start_idx) + 1), 2))

            for local_idx in np.ndindex(local_mask.shape[:-1]):
                local_end_idx = np.asarray(local_mask.shape[:-1]) - 1

                ampere_in = 0
                if local_idx[0] > 0:
                    ampere_in += local_mask[local_idx[0]-1, local_idx[1], 0]
                if local_idx[1] > 0:
                    ampere_in += local_mask[local_idx[0], local_idx[1]-1, 1]
                if ampere_in == 0:
                    ampere_in = 1

                if not np.all(local_idx == local_end_idx):
                    local_mask[local_idx[0], local_idx[1], :] = ampere_in*((local_end_idx - local_idx)/np.sum(local_end_idx - local_idx))

            posneg = np.sign(stop_idx - start_idx)

            if np.all(posneg == [-1, -1]):
                local_mask *= -1
                current_model[0, stop_idx[0]:start_idx[0]+1, stop_idx[1]:start_idx[1]+1, 1:] += ampere*local_mask
            elif np.all(posneg == [-1, 1]) or np.all(posneg == [-1, 0]):
                local_mask = np.flipud(local_mask)
                local_mask[..., 0] = np.roll(local_mask[..., 0], -1, axis=0)
                local_mask[..., 0] *= -1
                current_model[0, stop_idx[0]:start_idx[0]+1, start_idx[1]:stop_idx[1]+1, 1:] += ampere*local_mask
            elif np.all(posneg == [1, -1]) or np.all(posneg == [0, -1]):
                local_mask = np.fliplr(local_mask)
                local_mask[..., 1] = np.roll(local_mask[..., 1], -1, axis=1)
                local_mask[..., 1] *= -1
                current_model[0, start_idx[0]:stop_idx[0]+1, stop_idx[1]:start_idx[1]+1, 1:] += ampere*local_mask
            else:
                current_model[0, start_idx[0]:stop_idx[0]+1, start_idx[1]:stop_idx[1]+1, 1:] += ampere*local_mask
                
        # TODO: Test model with kirchoff current law
        # K = kirchoff_matrix(self.shape)
        # idxs_flattened = np.nonzero(K.dot(current_model.flatten()))
        # idxs = np.unravel_index(idxs_flattened, self.shape)
        # print('Warning: The model ADD_MODEL_NAME breaks Kirchoffs current law at positions:\n')
        # for ii in range(len(idxs_flattened))
        #    print([self._radgrid[idxs], self._thetagrid[idxs], self._phigrid[idxs]])
        
        self._current_models.update({name: current_model})


class GICi3DRegressor():

    def __init__(self, gici3d) -> None:
        self._gici3d = gici3d
        
        self._calculate_svd = False
        self._mag_pos = None
        self._Tm = None
        self._current_est = None
        self._u = None
        self._s = None
        self._vh = None

    @property
    def current_est(self):
        return self._current_est

    @property
    def mag_pos(self):
        return self._mag_pos

    @property
    def Tm(self):
        return self._Tm

    def get_svd_results(self):
        return [self._u, self._s, self._vh]

    def fit(self, mag_pos, mag_obs, alpha=1e-10):
        mag_pos = np.asarray(mag_pos)
        mag_obs = np.asarray(mag_obs)
        
        if mag_pos is not self._mag_pos:
            
            self._mag_pos = mag_pos
            self._transformation_matrix()
            self._calculate_svd = True

        K = kirchoff_matrix(self._gici3d.shape)
        _Tm = np.vstack([self._Tm, K.toarray()])
        _mag_obs = np.hstack([mag_obs.flatten(), np.zeros(K.shape[0])])

        # TODO: Implement and optimise use of sparse svd for large arrays that the normal svd cant handle 
        # T_kirchoff = sparse.vstack([sparse.csr_array(self._Tm), K])
        
        if self._calculate_svd:
            self._u, self._s, self._vh = la.svd(_Tm, full_matrices=False)

            self._calculate_svd = False

            # TODO: Implement and optimise use of sparse svd for large arrays that the normal svd cant handle 
            # if sparse.issparse(_T):
            #    self._u, self._s, self._vh = sparsela.svds(T, k=min(T.shape) - 1)
            # else:
            #    self._u, self._s, self._vh = la.svd(_T, full_matrices=False)

        if alpha == 0:
            # TODO: Implement fix that just uses nonzero singular values for the case of alpha=0
            # Throwing error when alpha=0, temporary until fix is implemented
            raise ValueError('Value alpha must be larger than zero')
        
        d = self._s/(self._s**2 + alpha**2)

        # self._current_est = np.dot(np.dot(self._vh.T, np.dot(np.diag(d), self._u.T)), _mag_obs)
        self._current_est = np.linalg.multi_dot([self._vh.T, np.diag(d), self._u.T, _mag_obs])

        return self

    def predict_mag_from_model(self, mag_pos, model_name):
        mag_pos = np.asarray(mag_pos)
        if mag_pos is not self._mag_pos:
            self._mag_pos = mag_pos
            self._transformation_matrix()
            self._calculate_svd = True
        
        current_model = self._gici3d.get_current_model(model_name)

        mag_obs = np.dot(self._Tm, current_model.flatten()).reshape(-1, 3)

        return mag_obs

    def _transformation_matrix(self):
        rgrid = self._gici3d.radgrid
        tgrid = self._gici3d.thetagrid
        pgrid = self._gici3d.phigrid

        args = []
        for idx in np.ndindex(rgrid.shape + (3,)):
            ridx = idx[0]
            tidx = idx[1]
            pidx = idx[2]

            r0 = np.array([rgrid[ridx, tidx, pidx], tgrid[ridx, tidx, pidx], pgrid[ridx, tidx, pidx]])

            if (idx[3] == 0) and (ridx != (rgrid.shape[0] - 1)):
                rad = rgrid[ridx+1, tidx, pidx]
                theta = tgrid[ridx+1, tidx, pidx]
                phi = pgrid[ridx+1, tidx, pidx]
            elif (idx[3] == 0) and (ridx == (rgrid.shape[0] - 1)):
                rad = rgrid[ridx, tidx, pidx] + (rgrid[ridx, tidx, pidx] - rgrid[ridx-1, tidx, pidx])
                theta = tgrid[ridx, tidx, pidx] + (tgrid[ridx, tidx, pidx] - tgrid[ridx-1, tidx, pidx])
                phi = pgrid[ridx, tidx, pidx] + (pgrid[ridx, tidx, pidx] - pgrid[ridx-1, tidx, pidx])
            elif (idx[3] == 1) and (tidx != (tgrid.shape[1] - 1)):
                rad = rgrid[ridx, tidx+1, pidx]
                theta = tgrid[ridx, tidx+1, pidx]
                phi = pgrid[ridx, tidx+1, pidx]
            elif (idx[3] == 1) and (tidx == (tgrid.shape[1] - 1)):
                rad = rgrid[ridx, tidx, pidx] + (rgrid[ridx, tidx, pidx] - rgrid[ridx, tidx-1, pidx])
                theta = tgrid[ridx, tidx, pidx] + (tgrid[ridx, tidx, pidx] - tgrid[ridx, tidx-1, pidx])
                phi = pgrid[ridx, tidx, pidx] + (pgrid[ridx, tidx, pidx] - pgrid[ridx, tidx-1, pidx])
            elif (idx[3] == 2) and (pidx != (pgrid.shape[2] - 1)):
                rad = rgrid[ridx, tidx, pidx+1]
                theta = tgrid[ridx, tidx, pidx+1]
                phi = pgrid[ridx, tidx, pidx+1]
            elif (idx[3] == 2) and (pidx == (pgrid.shape[2] - 1)):
                rad = rgrid[ridx, tidx, pidx] + (rgrid[ridx, tidx, pidx] - rgrid[ridx, tidx, pidx-1])
                theta = tgrid[ridx, tidx, pidx] + (tgrid[ridx, tidx, pidx] - tgrid[ridx, tidx, pidx-1])
                phi = pgrid[ridx, tidx, pidx] + (pgrid[ridx, tidx, pidx] - pgrid[ridx, tidx, pidx-1])
        
            r1 = np.array([rad, theta, phi])

            args.append([self._mag_pos, r0, r1, 1])

        pool = mp.Pool(processes=mp.cpu_count()-1)
        result = pool.starmap(biot_savart, args)
        pool.close()
        pool.join()

        self._Tm = np.array(result).reshape(-1, self._mag_pos.size).T

        return self
