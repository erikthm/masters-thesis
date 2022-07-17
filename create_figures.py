# %%
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from gici3d import GICi3D, GICi3DRegressor
from EMequations import biot_savart

radE = 6371.2

rad_grid = [radE + 100, radE + 200, 2]
theta_grid = [15, 30, 6]
phi_grid = [0, 30, 11]

date = datetime.now()

ionosphere_grid = GICi3D(rad_grid, theta_grid, phi_grid, date)

[radgrid, thetagrid, phigrid] = [ionosphere_grid.radgrid, ionosphere_grid.thetagrid, ionosphere_grid.phigrid]

currents = [['FAC_in', [18, 9], 1],
            ['horizontal', [18, 9], [27, 30], 1],
            ['FAC_out', [27, 30], 1]]

ionosphere_grid.add_current_model(currents, name='example_model')

current_model = ionosphere_grid.get_current_model('example_model')

tmp1 = np.insert(current_model[0, ..., 1], 0, 0, axis=0)
tmp1 = tmp1[1:, :]*(tmp1[1:, :] >= 0) + tmp1[:-1, :]*(tmp1[:-1, :] < 0)
tmp2 = np.insert(current_model[0, ..., 2], 0, 0, axis=1)
tmp2 = tmp2[:, 1:]*(tmp2[:, 1:] >= 0) + tmp2[:, :-1]*(tmp2[:, :-1] < 0)

title_fontsize = 35
label_fontsize = 25
tick_labelsize = 20

fig1 = plt.figure(figsize=(20, 20))
ax = fig1.add_subplot(projection='3d')
ax.scatter(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...], c='r', alpha=1)
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          np.ones(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape),
          length=2.7)
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          np.zeros(phigrid[0, ...].shape), np.ones(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape),
          length=2.7)
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          np.zeros(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape), np.ones(phigrid[0, ...].shape),
          length=200, arrow_length_ratio=0)

y_min, y_max = ax.get_ylim()
ax.set_ylim(y_max, y_min)
ax.set_xlabel('\n' + 'Longitude [deg]', linespacing=2, fontsize=label_fontsize)
ax.set_ylabel('\n' + 'Co-latitude [deg]', linespacing=2, fontsize=label_fontsize)
ax.set_zlabel('\n' + 'Radius [km]', linespacing=4, fontsize=label_fontsize)
ax.set_xticks(np.linspace(phi_grid[0], phi_grid[1], phi_grid[2]))
ax.set_yticks(np.linspace(theta_grid[0], theta_grid[1], theta_grid[2]))
ax.tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)

ax.set_title('GICi3D Wireframe Grid', fontsize=title_fontsize)
ax.view_init(25, -70)

fig2 = plt.figure(figsize=(20, 20))
ax = fig2.add_subplot(projection='3d', computed_zorder=False)
ax.scatter(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...], c='r', alpha=1, zorder=-1)
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          current_model[0, ..., 2], np.zeros(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape),
          length=3)
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          np.zeros(phigrid[0, ...].shape), current_model[0, ..., 1], np.zeros(phigrid[0, ...].shape),
          length=3)
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          np.zeros(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape), current_model[0, ..., 0]*(current_model[0, ..., 0]>=0),
          length=150, arrow_length_ratio=0, color='g')
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          np.zeros(phigrid[0, ...].shape), np.zeros(phigrid[0, ...].shape), current_model[0, ..., 0]*(current_model[0, ..., 0]<0),
          length=150, arrow_length_ratio=0,  pivot='tip', color='g')
ax.quiver(phigrid[0, ...], thetagrid[0, ...], radgrid[0, ...],
          (phigrid[0, current_model[0, ..., 0]>0]-phigrid[0, current_model[0, ..., 0]<0])*(current_model[0, ..., 0]<0), 
          (thetagrid[0, current_model[0, ..., 0]>0]-thetagrid[0, current_model[0, ..., 0]<0])*(current_model[0, ..., 0]<0),
          np.zeros(phigrid[0, ...].shape),
          length=1, arrow_length_ratio=.1, color='g')

y_min, y_max = ax.get_ylim()
ax.set_ylim(y_max, y_min)
ax.set_xlabel('\n' + 'Longitude [deg]', linespacing=2, fontsize=label_fontsize)
ax.set_ylabel('\n' + 'Co-latitude [deg]', linespacing=2, fontsize=label_fontsize)
ax.set_zlabel('\n' + 'Radius [km]', linespacing=4, fontsize=label_fontsize)
ax.set_xticks(np.linspace(phi_grid[0], phi_grid[1], phi_grid[2]))
ax.set_yticks(np.linspace(theta_grid[0], theta_grid[1], theta_grid[2]))
ax.tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)

ax.set_title('GICi3D Representation of a Simple Current', fontsize=title_fontsize)
ax.view_init(25, -70)

title_fontsize = 35
label_fontsize = 30
tick_labelsize = 25

fig3, ax = plt.subplots(1, 1, figsize=[30, 15])
quiv = ax.quiver(phigrid[0, ...], thetagrid[0, ...], tmp2, tmp1, tmp2+tmp1, angles='xy', scale=10, headwidth=2)
ax.scatter(phigrid[0, current_model[0, ..., 0] > 0], thetagrid[0, current_model[0, ..., 0] > 0],
           s=1000, color='r', marker='o', facecolors='none')
ax.scatter(phigrid[0, current_model[0, ..., 0] < 0], thetagrid[0, current_model[0, ..., 0] < 0],
           s=1000, color='r', marker='x')

ax.invert_yaxis()
ax.set_xlabel('Longitude [deg]', fontsize=label_fontsize)
ax.set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
ax.set_xticks(np.linspace(phi_grid[0], phi_grid[1], phi_grid[2]))
ax.set_yticks(np.linspace(theta_grid[0], theta_grid[1], theta_grid[2]))
ax.tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)

cbar = fig3.colorbar(quiv, ax=ax)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

ax.set_title('GICi3D Vector Field Representation of a Simple Current', fontsize=title_fontsize)

fig1.savefig(fname='Figures/gici3d_grid_example.jpg', bbox_inches='tight')
fig2.savefig(fname='Figures/gici3d_current_example.jpg', bbox_inches='tight')
fig3.savefig(fname='Figures/gici3d_vector_example.jpg', bbox_inches='tight')
# %%
