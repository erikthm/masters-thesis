import numpy as np
from scipy import sparse
from ppigrf import ppigrf

d2r = np.deg2rad
r2d = np.rad2deg


def kirchoff_matrix(grid_shape):

    i, j, k = grid_shape
    l = 3

    ijkl = i*j*k*l
    ijk = i*j*k
    jkl = j*k*l
    jk = j*k
    kl = k*l

    nodes = np.arange(ijk)

    thin = np.in1d(nodes % jk, np.arange(0, k, 1))
    phin = np.in1d(nodes % jk, np.arange(0, jk, k))
    thout = np.in1d(nodes % jk, np.arange(jk - k, jk, 1))
    phout = np.in1d(nodes % jk, np.arange(k-1, jk, k))

    mask = np.ones([len(nodes), 6], dtype=bool)
    mask[:jk, 0] = False
    mask[thin, 1] = False
    mask[phin, 2] = False
    mask[thout, 4] = False
    mask[phout, 5] = False
    mask = mask.flatten()

    data = np.tile([-1, -1, -1, 1, 1, 1], ijk)
    rows = np.repeat(nodes, 6)
    cols = rows*l
    cols[0::6] += 0 - jkl
    cols[1::6] += 1 - kl
    cols[2::6] += 2 - l
    cols[3::6] += 0
    cols[4::6] += 1
    cols[5::6] += 2

    data = data[mask]
    rows = rows[mask]
    cols = cols[mask]

    kirchoff_matrix = sparse.csr_array((data, (rows, cols)), shape=(ijk, ijkl))

    return kirchoff_matrix


def biot_savart(r_obs, r_start, r_stop, i_magnitude, coord_sys='sph'):
    """Calculates the magnetic field, at observation points r_obs, generated by a constant
    uniform current going between two points (r_start, r_stop) using the Biot-Savart law

    Args:
        r0 (_type_): _description_
        r1 (_type_): _description_
        r_obs (_type_): _description_
        current_magnitude (_type_): _description_

    Returns:
        _type_: _description_
    """

    if (r_obs.ndim not in (1, 2)) and (r_obs.shape[-1] != 3):
        # TODO: Throw Error
        pass

    if r_obs.ndim == 1:
        r_obs = r_obs.reshape(1, 3)
    
    def cross_product(dl, r_prime):
        return np.array([dl[:, 1]*r_prime[:, :, 2] - dl[:, 2]*r_prime[:, :, 1],
                         dl[:, 2]*r_prime[:, :, 0] - dl[:, 0]*r_prime[:, :, 2],
                         dl[:, 0]*r_prime[:, :, 1] - dl[:, 1]*r_prime[:, :, 0]])

    mu_0 = 4*np.pi*1e-7

    steps = int(np.round(np.linalg.norm(r_stop - r_start))/10)
    if steps < 100:
        steps = 100

    dr = (r_stop - r_start)*np.expand_dims(np.linspace(1/steps, 1, steps), axis=1)
    if coord_sys == 'sph':
        dl = np.diff(sph2cart(r_start + dr) - sph2cart(r_start), axis=0, prepend=0)
        r_prime = np.expand_dims(sph2cart(r_obs), axis=1) - sph2cart(r_start + dr)
    elif coord_sys == 'cart':
        dl = np.diff(dr, axis=0, prepend=0)
        r_prime = np.expand_dims(r_obs, axis=1) - (r_start + dr)

    solved_integral = np.sum(cross_product(dl, r_prime)/(np.linalg.norm(r_prime, axis=2)**3), axis=-1)

    B = (i_magnitude*mu_0/(4*np.pi))*solved_integral

    return B.T


def trace_mag_field(rad0, theta0, phi0, radmax, stepsize, date):
    rad = np.asarray([rad0]).reshape(1, np.asarray(rad0).size)
    theta = np.asarray([theta0]).reshape(1, np.asarray(theta0).size)
    phi = np.asarray([phi0]).reshape(1, np.asarray(phi0).size)

    ii = 0
    while rad[ii, :].min() <= radmax:
        magfield = ppigrf.igrf_gc(rad[ii, :], theta[ii, :], phi[ii, :], date)
        magfield = np.concatenate(magfield, axis=0)
        magfield[:, theta[ii, :] < 90] = -magfield

        step_dir = stepsize*magfield/np.linalg.norm(magfield, axis=0)

        newrad = rad[ii, :] + step_dir[0, :]
        newtheta = d2r(theta[ii, :]) + step_dir[1, :]/rad[ii, :]
        newphi = d2r(phi[ii, :]) + step_dir[2, :]/(rad[ii, :]*np.sin(d2r(theta[ii, :])))

        rad = np.vstack([rad, newrad])
        theta = np.vstack([theta, r2d(newtheta)])
        phi = np.vstack([phi, r2d(newphi)])

        if rad[ii+1, :].min() <= rad[ii, :].min():
            break

        ii += 1

    return [rad, theta, phi]


def sph2cart(r_theta_phi, deg=True):
    r, theta, phi = np.hsplit(r_theta_phi, 3)

    if deg:
        theta = d2r(theta)
        phi = d2r(phi)
    
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return np.column_stack((x, y, z))
    