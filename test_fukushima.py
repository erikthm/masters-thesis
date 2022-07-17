# %%
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation

from EMequations import biot_savart, trace_mag_field

rng = np.random.default_rng()

radE = 6371.2
ampere = 1e4

mag_pos = np.column_stack([radE*np.ones(50), rng.random(50)*10 + 15, rng.random(50)*10 + 15])
# mag_pos = np.column_stack([radE*np.ones(50), rng.random(50)*10 + 65, rng.random(50)*10 + 15])

rad_traced, theta_traced, phi_traced = trace_mag_field(radE + 100, 20, 20, radE + 1000, 10, datetime.now())
# rad_traced, theta_traced, phi_traced = trace_mag_field(radE + 100, 70, 20, radE + 1000, 10, datetime.now())

I_fac = np.column_stack([rad_traced, theta_traced, phi_traced])
I_vert = np.column_stack([rad_traced,
                          np.repeat(theta_traced[0], I_fac.shape[0]),
                          np.repeat(phi_traced[0], I_fac.shape[0])])

vector = [0, 0, 20]

rot_deg = 1

rot = Rotation.from_rotvec(np.column_stack([np.arange(0, 2*np.pi, (rot_deg/180)*np.pi),
                                            np.zeros(int(360/rot_deg)),
                                            np.zeros(int(360/rot_deg))]))

rot_vector = rot.apply(vector)

I_div = np.broadcast_to(np.array([radE + 100, 20, 20]), (rot_vector.shape[0], 2, 3)).copy()
# I_div = np.broadcast_to(np.array([radE + 100, 70, 20]), (rot_vector.shape[0], 2, 3)).copy()
I_div[:, 1, :] += rot.apply(vector)

mag_obs_div = 0
for ii in range(I_div.shape[0]):
    mag_obs_div += biot_savart(mag_pos, I_div[ii, 0, :], I_div[ii, 1, :], ampere/I_div.shape[0])

mag_obs_vert = 0
for ii in range(1, I_vert.shape[0]):
    mag_obs_vert += biot_savart(mag_pos, I_vert[ii-1, :], I_vert[ii, :], ampere)

mag_obs_fac = 0
for ii in range(1, I_fac.shape[0]):
    mag_obs_fac += biot_savart(mag_pos, I_fac[ii-1, :], I_fac[ii, :], ampere)


mag_obs_fukushima = mag_obs_vert + mag_obs_div
mag_obs_real = mag_obs_fac + mag_obs_div

mag_obs_diff = np.abs(mag_obs_fukushima - mag_obs_real)

title_fontsize = 25
label_fontsize = 20
tick_labelsize = 15

fig1 = plt.figure(figsize=(20, 20))
ax = fig1.add_subplot(projection='3d')
label_added = False
for ii in range(0, int(I_div.shape[0]), 10):
    if not label_added:
        ax.plot3D(I_div[ii, :, 2], I_div[ii, :, 1], I_div[ii, :, 0], color='b', label='Horizontal currents')
        label_added = True
    else:
        ax.plot3D(I_div[ii, :, 2], I_div[ii, :, 1], I_div[ii, :, 0], color='b')
ax.plot3D(I_vert[:, 2], I_vert[:, 1], I_vert[:, 0], color='r', label='Vertical current (Fukushima)')
ax.plot3D(I_fac[:, 2], I_fac[:, 1], I_fac[:, 0], color='g', label='Field Aligned Current')
ax.scatter(mag_pos[:, 2], mag_pos[:, 1], mag_pos[:, 0], color='k', alpha=1)

y_min, y_max = ax.get_ylim()
ax.set_ylim(y_max, y_min)
ax.set_xlabel('\n' + 'Longitude [deg]', linespacing=2, fontsize=label_fontsize)
ax.set_ylabel('\n' + 'Co-latitude [deg]', linespacing=2, fontsize=label_fontsize)
ax.set_zlabel('\n' + 'Radius [km]', linespacing=2, fontsize=label_fontsize)
ax.tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)

ax.set_title('Downward vertical current, FAC and diverging horizontal currents\n' +
             'above a set of magnetic measurement stations', fontsize=title_fontsize)

ax.legend(fontsize=label_fontsize)
ax.view_init(25, -25)

title_fontsize = 30
label_fontsize = 25
tick_labelsize = 18

fig2, ax = plt.subplots(3, 1, figsize=(20, 20), constrained_layout=True, sharex=True)
for ii in range(len(ax)):
    # Plotting data
    ax[ii].scatter(np.arange(1, mag_obs_fukushima.shape[0]+1), mag_obs_fukushima[:, ii]*1e9, color='r')
    ax[ii].scatter(np.arange(1, mag_obs_real.shape[0]+1), mag_obs_real[:, ii]*1e9, color='b')

    # Adjusting style
    ax[ii].grid(True, axis='x')
    ax[ii].set_axisbelow(True)
    ax[ii].set_xticks(np.arange(1, mag_pos.shape[0]+1))
    ax[ii].tick_params(which='major', length=10, width=2, labelsize=tick_labelsize)
    ax[ii].axhline(0, linestyle='--', color='gray', linewidth=0.5)
    y_max = np.abs(ax[ii].get_ylim()).max()
    ax[ii].set_ylim(-y_max, y_max)

ax[0].set_ylabel(r'B$_{r}$ [nT]', fontsize=label_fontsize)
ax[1].set_ylabel(r'B$_{\theta}$ [nT]', fontsize=label_fontsize)
ax[2].set_ylabel(r'B$_{\phi}$ [nT]', fontsize=label_fontsize)
ax[2].set_xlabel('Station number', fontsize=label_fontsize)
ax[0].legend(['Vertical current (Fukushima)', 'Field Aligned Current'], fontsize=label_fontsize)
ax[0].set_title('Ground magnetic field measurements from vertical and FAC current systems', fontsize=title_fontsize)

fig3, ax = plt.subplots(3, 1, figsize=(20, 20), constrained_layout=True, sharex=True)
for ii in range(len(ax)):
    ax[ii].scatter(np.arange(1, mag_obs_diff.shape[0]+1), mag_obs_diff[:, ii]*1e9, color='k')
    
    ax[ii].grid(True, axis='x')
    ax[ii].set_axisbelow(True)
    ax[ii].set_xticks(np.arange(1, mag_pos.shape[0]+1))
    ax[ii].tick_params(which='major', length=10, width=2, labelsize=tick_labelsize)
    y_max = np.abs(ax[ii].get_ylim()).max()
    ax[ii].set_ylim(ymin=0, ymax=y_max)

ax[0].set_ylabel(r'B$_{r}$ [nT]', fontsize=label_fontsize)
ax[1].set_ylabel(r'B$_{\theta}$ [nT]', fontsize=label_fontsize)
ax[2].set_ylabel(r'B$_{\phi}$ [nT]', fontsize=label_fontsize)
ax[2].set_xlabel('Station number', fontsize=label_fontsize)
ax[0].set_title('Difference in magnetic field measurements between vertical and FAC current systems', 
                fontsize=title_fontsize)

if I_vert[0, 1] == 70:
    fig1.savefig(fname='Figures/fukushima_system70.jpg', bbox_inches='tight')
    fig2.savefig(fname='Figures/fukushima_magvertfac70.jpg', bbox_inches='tight')
    fig3.savefig(fname='Figures/fukushima_magdiff70.jpg', bbox_inches='tight')
elif I_vert[0, 1] == 20:
    fig1.savefig(fname='Figures/fukushima_system20.jpg', bbox_inches='tight')
    fig2.savefig(fname='Figures/fukushima_magvertfac20.jpg', bbox_inches='tight')
    fig3.savefig(fname='Figures/fukushima_magdiff20.jpg', bbox_inches='tight')
# %%
