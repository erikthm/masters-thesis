#!env/bin/python
# %%
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from gici3d import GICi3D, GICi3DRegressor

from current_systems import (east_halfdeg, east_1deg, east_3deg, east_6deg, east_9deg, east_12deg, east_15deg,
                             south_halfdeg, south_1deg, south_2deg, south_4deg, south_6deg, south_8deg, south_10deg,
                             eastwardstream, southwardstream, multidirection)

rng = np.random.default_rng()

radE = 6371.2

rad_grid = [radE + 100, radE + 200, 2]
theta_grid = [20, 30, 21]
phi_grid = [15, 30, 31]

date = datetime.now()

ionosphere_grid = GICi3D(rad_grid, theta_grid, phi_grid, date)

[radgrid, thetagrid, phigrid] = [ionosphere_grid.radgrid, ionosphere_grid.thetagrid, ionosphere_grid.phigrid]

ionosphere_grid.add_current_model(east_halfdeg, name='east_halfdeg')
ionosphere_grid.add_current_model(east_1deg, name='east_1deg')
ionosphere_grid.add_current_model(east_3deg, name='east_3deg')
ionosphere_grid.add_current_model(east_6deg, name='east_6deg')
ionosphere_grid.add_current_model(east_9deg, name='east_9deg')
ionosphere_grid.add_current_model(east_12deg, name='east_12deg')
ionosphere_grid.add_current_model(east_15deg, name='east_15deg')
ionosphere_grid.add_current_model(south_halfdeg, name='south_halfdeg')
ionosphere_grid.add_current_model(south_1deg, name='south_1deg')
ionosphere_grid.add_current_model(south_2deg, name='south_2deg')
ionosphere_grid.add_current_model(south_4deg, name='south_4deg')
ionosphere_grid.add_current_model(south_6deg, name='south_6deg')
ionosphere_grid.add_current_model(south_8deg, name='south_8deg')
ionosphere_grid.add_current_model(south_10deg, name='south_10deg')
ionosphere_grid.add_current_model(eastwardstream, name='eastwardstream')
ionosphere_grid.add_current_model(southwardstream, name='southwardstream')
ionosphere_grid.add_current_model(multidirection, name='multidirection')

regressor = GICi3DRegressor(ionosphere_grid)

# mag_pos = np.column_stack([radE*np.ones(50), rng.random(50)*10 + 20, rng.random(50)*15 + 15])
# mag_pos = np.column_stack([radE*np.ones(100), rng.random(100)*10 + 20, rng.random(100)*15 + 15])
# mag_pos = np.column_stack([radE*np.ones(200), rng.random(200)*10 + 20, rng.random(200)*15 + 15])
mag_pos = np.column_stack([radE*np.ones(500), rng.random(500)*10 + 20, rng.random(500)*15 + 15])

figs = {}

title_fontsize = 35
label_fontsize = 30
tick_labelsize = 25

fig, ax = plt.subplots(1, 1, figsize=[30, 15])
ax.scatter(mag_pos[:, 2], mag_pos[:, 1])

ax.invert_yaxis()
ax.set_xlabel('Longitude [deg]', fontsize=label_fontsize)
ax.set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
ax.set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax.set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax.tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax.tick_params(which='minor', length=7, width=2)
ax.xaxis.set_minor_locator(AutoMinorLocator(6))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.grid()

ax.set_title('Ground Magnetic Measurement Locations', fontsize=title_fontsize)

figs.update({f'{mag_pos.shape[0]}magpos_locations': fig})

east_systems_names = ['east_halfdeg', 'east_1deg', 'east_3deg', 'east_6deg',
                      'east_9deg', 'east_12deg', 'east_15deg', 'eastwardstream']
south_systems_names = ['south_halfdeg', 'south_1deg', 'south_2deg', 'south_4deg',
                       'south_6deg', 'south_8deg', 'south_10deg', 'southwardstream']

for system_name in east_systems_names:
    actual_system = ionosphere_grid.get_current_model(system_name)
    mag_obs = regressor.predict_mag_from_model(mag_pos, system_name)

    mae = np.ones(50)
    alpha = np.ones(50)
    alpha[1] = 0.5
    for ii in range(len(mae)):
        if ii > 1:
            alpha[ii] = alpha[ii-2]*0.1

        regressor.fit(mag_pos, mag_obs, alpha=alpha[ii])
        predicted_system = regressor.current_est

        predicted_system = predicted_system.reshape(actual_system.shape)

        # Normalizing data to the range [0, 1]
        actual_norm = (actual_system - actual_system.min())/(actual_system.max() - actual_system.min())
        predicted_norm = (predicted_system - predicted_system.min())/(predicted_system.max() - predicted_system.min())

        mae[ii] = np.mean(np.abs(predicted_norm-actual_norm))

    print(f'The best prediction of the model {system_name}'
          f' has mean absolute error of {mae.min():.2E} with regularization parameter {alpha[mae.argmin()]:.2E}')

    regressor.fit(mag_pos, mag_obs, alpha=alpha[mae.argmin()])
    predicted_system = regressor.current_est
    predicted_system = predicted_system.reshape(actual_system.shape)

    # TODO: Change names of tmp1, tmp2
    tmp1 = np.insert(actual_system[0, ..., 1], 0, 0, axis=0)
    tmp1 = tmp1[1:, :]*(tmp1[1:, :] >= 0) + tmp1[:-1, :]*(tmp1[:-1, :] < 0)
    tmp2 = np.insert(actual_system[0, ..., 2], 0, 0, axis=1)
    tmp2 = tmp2[:, 1:]*(tmp2[:, 1:] >= 0) + tmp2[:, :-1]*(tmp2[:, :-1] < 0)

    tmp1_est = np.insert(predicted_system[0, ..., 1], 0, 0, axis=0)
    tmp1_est = tmp1_est[1:, :]*(tmp1_est[1:, :] >= 0) + tmp1_est[:-1, :]*(tmp1_est[:-1, :] < 0)
    tmp2_est = np.insert(predicted_system[0, ..., 2], 0, 0, axis=1)
    tmp2_est = tmp2_est[:, 1:]*(tmp2_est[:, 1:] >= 0) + tmp2_est[:, :-1]*(tmp2_est[:, :-1] < 0)

    title_fontsize = 35
    label_fontsize = 30
    tick_labelsize = 25

    fig, ax = plt.subplots(2, 1, figsize=[30, 30])
    quiv = ax[0].quiver(phigrid[0, ...], thetagrid[0, ...], tmp2, tmp1, tmp2+tmp1,
                        cmap='seismic', angles='xy', scale_units='xy', scale=2.2e4,
                        headlength=0, headwidth=0, headaxislength=0)
    sc1 = ax[0].scatter(phigrid[0, actual_system[0, ..., 0] > 0],
                        thetagrid[0, actual_system[0, ..., 0] > 0],
                        s=0.05*np.abs(actual_system[0, actual_system[0, ..., 0] > 0, 0]),
                        color='k', marker='o', facecolors='none')
    sc2 = ax[0].scatter(phigrid[0, actual_system[0, ..., 0] < 0],
                        thetagrid[0, actual_system[0, ..., 0] < 0],
                        s=0.05*np.abs(actual_system[0, actual_system[0, ..., 0] < 0, 0]),
                        color='k', marker='x')
    
    quiv.set_clim(-1e4, 1e4)
    
    ax[0].set_title('Actual Current System', fontsize=title_fontsize)

    ax[0].invert_yaxis()
    ax[0].set_xlabel('Longitude [deg]', fontsize=label_fontsize)
    ax[0].set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
    ax[0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0].tick_params(which='minor', length=7, width=2)
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))

    lg1 = ax[0].legend(*sc1.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='upper left',
                       title='Upward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    lg2 = ax[0].legend(*sc2.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='lower left',
                       title='Downward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    ax[0].add_artist(lg1)

    quiv = ax[1].quiver(phigrid[0, ...], thetagrid[0, ...], tmp2_est, tmp1_est, tmp2_est+tmp1_est,
                        cmap='seismic', angles='xy', scale_units='xy', scale=2.2e4,
                        headlength=0, headwidth=0, headaxislength=0, zorder=2)
    sc1 = ax[1].scatter(phigrid[0, predicted_system[0, ..., 0] > 0],
                        thetagrid[0, predicted_system[0, ..., 0] > 0],
                        s=0.05*np.abs(predicted_system[0, predicted_system[0, ..., 0] > 0, 0]),
                        color='k', marker='o', facecolors='none', zorder=1)
    sc2 = ax[1].scatter(phigrid[0, predicted_system[0, ..., 0] < 0],
                        thetagrid[0, predicted_system[0, ..., 0] < 0],
                        s=0.05*np.abs(predicted_system[0, predicted_system[0, ..., 0] < 0, 0]),
                        color='k', marker='x', zorder=1)
    
    quiv.set_clim(-1e4, 1e4)

    ax[1].set_title('Predicted Current System', fontsize=title_fontsize)
    
    ax[1].invert_yaxis()
    ax[1].minorticks_on()
    ax[1].set_xlabel('Longitude [deg]', fontsize=label_fontsize)
    ax[1].set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
    ax[1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1].tick_params(which='minor', length=7, width=2)
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))

    lg1 = ax[1].legend(*sc1.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='upper left',
                       title='Upward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    lg2 = ax[1].legend(*sc2.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='lower left',
                       title='Downward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    ax[1].add_artist(lg1)

    cbar = fig.colorbar(quiv, ax=ax.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    figs.update({f'{mag_pos.shape[0]}magpos_{system_name}_vector': fig})

    title_fontsize = 35
    suptitle_fontsize = 40
    suplabel_fontsize = 30
    tick_labelsize = 25

    fig, ax = plt.subplots(2, 3, figsize=[30, 15], constrained_layout=True, sharex=True, sharey=True)
    im = ax[0, 0].imshow(actual_system[0, ..., 2],
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

    ax[0, 0].minorticks_on()
    ax[0, 0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0, 0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0, 0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0, 0].tick_params(which='minor', length=7, width=2)
    ax[0, 0].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0, 0].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[0, 0].set_title('Actual longitudinal currents', fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[1, 0].imshow(predicted_system[0, ..., 2],
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

    ax[1, 0].minorticks_on()
    ax[1, 0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1, 0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1, 0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1, 0].tick_params(which='minor', length=7, width=2)
    ax[1, 0].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1, 0].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[1, 0].set_title('Predicted longitudinal currents', fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax[1, 0], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[0, 1].imshow(actual_system[0, ..., 0]*(actual_system[0, ..., 0] < 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[0, 1].minorticks_on()
    ax[0, 1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0, 1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0, 1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0, 1].tick_params(which='minor', length=7, width=2)
    ax[0, 1].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0, 1].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[0, 1].set_title('Actual downward FACs', fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[1, 1].imshow(predicted_system[0, ..., 0]*(predicted_system[0, ..., 0] < 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[1, 1].minorticks_on()
    ax[1, 1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1, 1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1, 1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1, 1].tick_params(which='minor', length=7, width=2)
    ax[1, 1].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1, 1].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[1, 1].set_title('Predicted downward FACs', fontsize=title_fontsize)
    
    cbar = fig.colorbar(im, ax=ax[1, 1], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[0, 2].imshow(actual_system[0, ..., 0]*(actual_system[0, ..., 0] >= 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[0, 2].minorticks_on()
    ax[0, 2].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0, 2].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0, 2].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0, 2].tick_params(which='minor', length=7, width=2)
    ax[0, 2].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0, 2].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[0, 2].set_title('Actual upward FACs', fontsize=title_fontsize)
    
    cbar = fig.colorbar(im, ax=ax[0, 2], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[1, 2].imshow(predicted_system[0, ..., 0]*(predicted_system[0, ..., 0] >= 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[1, 2].minorticks_on()
    ax[1, 2].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1, 2].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1, 2].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1, 2].tick_params(which='minor', length=7, width=2)
    ax[1, 2].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1, 2].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[1, 2].set_title('Predicted upward FACs', fontsize=title_fontsize)
    
    cbar = fig.colorbar(im, ax=ax[1, 2], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    fig.supxlabel('Longitude [deg]', fontsize=suplabel_fontsize)
    fig.supylabel('Co-latitude [deg]', fontsize=suplabel_fontsize)
    fig.suptitle('Actual vs. Predicted Current Values in Grid', fontsize=suptitle_fontsize)

    figs.update({f'{mag_pos.shape[0]}magpos_{system_name}_values': fig})

for system_name in south_systems_names:
    actual_system = ionosphere_grid.get_current_model(system_name)
    mag_obs = regressor.predict_mag_from_model(mag_pos, system_name)

    mae = np.ones(50)
    alpha = np.ones(50)
    alpha[1] = 0.5
    for ii in range(len(mae)):
        if ii > 1:
            alpha[ii] = alpha[ii-2]*0.1

        regressor.fit(mag_pos, mag_obs, alpha=alpha[ii])
        predicted_system = regressor.current_est

        predicted_system = predicted_system.reshape(actual_system.shape)

        # Normalizing data to the range [0, 1]
        actual_norm = (actual_system - actual_system.min())/(actual_system.max() - actual_system.min())
        predicted_norm = (predicted_system - predicted_system.min())/(predicted_system.max() - predicted_system.min())

        mae[ii] = np.mean(np.abs(predicted_norm-actual_norm))

    print(f'The best prediction of the model {system_name}' +
          f' has mean absolute error of {mae.min():.2E} with regularization parameter {alpha[mae.argmin()]:.2E}')

    regressor.fit(mag_pos, mag_obs, alpha=alpha[mae.argmin()])
    predicted_system = regressor.current_est
    predicted_system = predicted_system.reshape(actual_system.shape)

    # TODO: Change names of tmp1, tmp2
    tmp1 = np.insert(actual_system[0, ..., 1], 0, 0, axis=0)
    tmp1 = tmp1[1:, :]*(tmp1[1:, :] >= 0) + tmp1[:-1, :]*(tmp1[:-1, :] < 0)
    tmp2 = np.insert(actual_system[0, ..., 2], 0, 0, axis=1)
    tmp2 = tmp2[:, 1:]*(tmp2[:, 1:] >= 0) + tmp2[:, :-1]*(tmp2[:, :-1] < 0)

    tmp1_est = np.insert(predicted_system[0, ..., 1], 0, 0, axis=0)
    tmp1_est = tmp1_est[1:, :]*(tmp1_est[1:, :] >= 0) + tmp1_est[:-1, :]*(tmp1_est[:-1, :] < 0)
    tmp2_est = np.insert(predicted_system[0, ..., 2], 0, 0, axis=1)
    tmp2_est = tmp2_est[:, 1:]*(tmp2_est[:, 1:] >= 0) + tmp2_est[:, :-1]*(tmp2_est[:, :-1] < 0)

    title_fontsize = 35
    label_fontsize = 25
    tick_labelsize = 25

    fig, ax = plt.subplots(2, 1, figsize=[30, 30])
    quiv = ax[0].quiver(phigrid[0, ...], thetagrid[0, ...], tmp2, tmp1, tmp2+tmp1,
                        cmap='seismic', angles='xy', scale_units='xy', scale=2.2e4,
                        headlength=0, headwidth=0, headaxislength=0)
    sc1 = ax[0].scatter(phigrid[0, actual_system[0, ..., 0] > 0],
                        thetagrid[0, actual_system[0, ..., 0] > 0],
                        s=0.05*np.abs(actual_system[0, actual_system[0, ..., 0] > 0, 0]),
                        color='k', marker='o', facecolors='none')
    sc2 = ax[0].scatter(phigrid[0, actual_system[0, ..., 0] < 0],
                        thetagrid[0, actual_system[0, ..., 0] < 0],
                        s=0.05*np.abs(actual_system[0, actual_system[0, ..., 0] < 0, 0]),
                        color='k', marker='x')
    
    quiv.set_clim(-1e4, 1e4)
    
    ax[0].set_title('Actual Current System', fontsize=title_fontsize)

    ax[0].invert_yaxis()
    ax[0].set_xlabel('Longitude [deg]', fontsize=label_fontsize)
    ax[0].set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
    ax[0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0].tick_params(which='minor', length=7, width=2)
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))

    lg1 = ax[0].legend(*sc1.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='upper left',
                       title='Upward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    lg2 = ax[0].legend(*sc2.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='lower left',
                       title='Downward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    ax[0].add_artist(lg1)

    quiv = ax[1].quiver(phigrid[0, ...], thetagrid[0, ...], tmp2_est, tmp1_est, tmp2_est+tmp1_est,
                        cmap='seismic', angles='xy', scale_units='xy', scale=2.2e4,
                        headlength=0, headwidth=0, headaxislength=0, zorder=2)
    sc1 = ax[1].scatter(phigrid[0, predicted_system[0, ..., 0] > 0],
                        thetagrid[0, predicted_system[0, ..., 0] > 0],
                        s=0.05*np.abs(predicted_system[0, predicted_system[0, ..., 0] > 0, 0]),
                        color='k', marker='o', facecolors='none', zorder=1)
    sc2 = ax[1].scatter(phigrid[0, predicted_system[0, ..., 0] < 0],
                        thetagrid[0, predicted_system[0, ..., 0] < 0],
                        s=0.05*np.abs(predicted_system[0, predicted_system[0, ..., 0] < 0, 0]),
                        color='k', marker='x', zorder=1)
    
    quiv.set_clim(-1e4, 1e4)

    ax[1].set_title('Predicted Current System', fontsize=title_fontsize)
    
    ax[1].invert_yaxis()
    ax[1].minorticks_on()
    ax[1].set_xlabel('Longitude [deg]', fontsize=label_fontsize)
    ax[1].set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
    ax[1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1].tick_params(which='minor', length=7, width=2)
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))

    lg1 = ax[1].legend(*sc1.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='upper left',
                       title='Upward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    lg2 = ax[1].legend(*sc2.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='lower left',
                       title='Downward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
    ax[1].add_artist(lg1)

    cbar = fig.colorbar(quiv, ax=ax.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    figs.update({f'{mag_pos.shape[0]}magpos_{system_name}_vector': fig})

    title_fontsize = 35
    suptitle_fontsize = 40
    suplabel_fontsize = 30
    tick_labelsize = 25

    fig, ax = plt.subplots(2, 3, figsize=[30, 15], constrained_layout=True, sharex=True, sharey=True)
    im = ax[0, 0].imshow(actual_system[0, ..., 1],
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

    ax[0, 0].minorticks_on()
    ax[0, 0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0, 0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0, 0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0, 0].tick_params(which='minor', length=7, width=2)
    ax[0, 0].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0, 0].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[0, 0].set_title('Actual co-latitudinal currents', fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax[0, 0], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[1, 0].imshow(predicted_system[0, ..., 1],
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

    ax[1, 0].minorticks_on()
    ax[1, 0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1, 0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1, 0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1, 0].tick_params(which='minor', length=7, width=2)
    ax[1, 0].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1, 0].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[1, 0].set_title('Predicted co-latitudinal currents', fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax[1, 0], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[0, 1].imshow(actual_system[0, ..., 0]*(actual_system[0, ..., 0] < 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[0, 1].minorticks_on()
    ax[0, 1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0, 1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0, 1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0, 1].tick_params(which='minor', length=7, width=2)
    ax[0, 1].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0, 1].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[0, 1].set_title('Actual downward FACs', fontsize=title_fontsize)

    cbar = fig.colorbar(im, ax=ax[0, 1], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[1, 1].imshow(predicted_system[0, ..., 0]*(predicted_system[0, ..., 0] < 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[1, 1].minorticks_on()
    ax[1, 1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1, 1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1, 1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1, 1].tick_params(which='minor', length=7, width=2)
    ax[1, 1].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1, 1].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[1, 1].set_title('Predicted downward FACs', fontsize=title_fontsize)
    
    cbar = fig.colorbar(im, ax=ax[1, 1], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[0, 2].imshow(actual_system[0, ..., 0]*(actual_system[0, ..., 0] >= 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[0, 2].minorticks_on()
    ax[0, 2].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[0, 2].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[0, 2].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[0, 2].tick_params(which='minor', length=7, width=2)
    ax[0, 2].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[0, 2].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[0, 2].set_title('Actual upward FACs', fontsize=title_fontsize)
    
    cbar = fig.colorbar(im, ax=ax[0, 2], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    im = ax[1, 2].imshow(predicted_system[0, ..., 0]*(predicted_system[0, ..., 0] >= 0),
                         extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])
    
    ax[1, 2].minorticks_on()
    ax[1, 2].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
    ax[1, 2].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
    ax[1, 2].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
    ax[1, 2].tick_params(which='minor', length=7, width=2)
    ax[1, 2].xaxis.set_minor_locator(AutoMinorLocator(6))
    ax[1, 2].yaxis.set_minor_locator(AutoMinorLocator(4))

    ax[1, 2].set_title('Predicted upward FACs', fontsize=title_fontsize)
    
    cbar = fig.colorbar(im, ax=ax[1, 2], shrink=0.7)
    cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    fig.supxlabel('Longitude [deg]', fontsize=suplabel_fontsize)
    fig.supylabel('Co-latitude [deg]', fontsize=suplabel_fontsize)
    fig.suptitle('Actual vs. Predicted Current Values in Grid', fontsize=suptitle_fontsize)

    figs.update({f'{mag_pos.shape[0]}magpos_{system_name}_values': fig})


actual_system = ionosphere_grid.get_current_model('multidirection')
mag_obs = regressor.predict_mag_from_model(mag_pos, 'multidirection')

mae = np.ones(50)
alpha = np.ones(50)
alpha[1] = 0.5
for ii in range(len(mae)):
    if ii > 1:
        alpha[ii] = alpha[ii-2]*0.1

    regressor.fit(mag_pos, mag_obs, alpha=alpha[ii])
    predicted_system = regressor.current_est

    predicted_system = predicted_system.reshape(actual_system.shape)

    # Normalizing data to the range [0, 1]
    actual_norm = (actual_system - actual_system.min())/(actual_system.max() - actual_system.min())
    predicted_norm = (predicted_system - predicted_system.min())/(predicted_system.max() - predicted_system.min())

    mae[ii] = np.mean(np.abs(predicted_norm-actual_norm))

print('The best prediction of the model multidirection' +
      f' has mean absolute error of {mae.min():.2E} with regularization parameter {alpha[mae.argmin()]:.2E}')

regressor.fit(mag_pos, mag_obs, alpha=alpha[mae.argmin()])
predicted_system = regressor.current_est
predicted_system = predicted_system.reshape(actual_system.shape)

# TODO: Change names of tmp1, tmp2
tmp1 = np.insert(actual_system[0, ..., 1], 0, 0, axis=0)
tmp1 = tmp1[1:, :]*(tmp1[1:, :] >= 0) + tmp1[:-1, :]*(tmp1[:-1, :] < 0)
tmp2 = np.insert(actual_system[0, ..., 2], 0, 0, axis=1)
tmp2 = tmp2[:, 1:]*(tmp2[:, 1:] >= 0) + tmp2[:, :-1]*(tmp2[:, :-1] < 0)

tmp1_est = np.insert(predicted_system[0, ..., 1], 0, 0, axis=0)
tmp1_est = tmp1_est[1:, :]*(tmp1_est[1:, :] >= 0) + tmp1_est[:-1, :]*(tmp1_est[:-1, :] < 0)
tmp2_est = np.insert(predicted_system[0, ..., 2], 0, 0, axis=1)
tmp2_est = tmp2_est[:, 1:]*(tmp2_est[:, 1:] >= 0) + tmp2_est[:, :-1]*(tmp2_est[:, :-1] < 0)

title_fontsize = 35
label_fontsize = 25
tick_labelsize = 25

fig, ax = plt.subplots(2, 1, figsize=[30, 30])
quiv = ax[0].quiver(phigrid[0, ...], thetagrid[0, ...], tmp2, tmp1, tmp2+tmp1,
                    cmap='seismic', angles='xy', scale_units='xy', scale=2.2e4,
                    headlength=0, headwidth=0, headaxislength=0)
sc1 = ax[0].scatter(phigrid[0, actual_system[0, ..., 0] > 0],
                    thetagrid[0, actual_system[0, ..., 0] > 0],
                    s=0.05*np.abs(actual_system[0, actual_system[0, ..., 0] > 0, 0]),
                    color='k', marker='o', facecolors='none')
sc2 = ax[0].scatter(phigrid[0, actual_system[0, ..., 0] < 0],
                    thetagrid[0, actual_system[0, ..., 0] < 0],
                    s=0.05*np.abs(actual_system[0, actual_system[0, ..., 0] < 0, 0]),
                    color='k', marker='x')

quiv.set_clim(-1e4, 1e4)

ax[0].set_title('Actual Current System', fontsize=title_fontsize)

ax[0].invert_yaxis()
ax[0].set_xlabel('Longitude [deg]', fontsize=label_fontsize)
ax[0].set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
ax[0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[0].tick_params(which='minor', length=7, width=2)
ax[0].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))

lg1 = ax[0].legend(*sc1.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='upper left',
                    title='Upward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
lg2 = ax[0].legend(*sc2.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='lower left',
                    title='Downward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
ax[0].add_artist(lg1)

quiv = ax[1].quiver(phigrid[0, ...], thetagrid[0, ...], tmp2_est, tmp1_est, tmp2_est+tmp1_est,
                    cmap='seismic', angles='xy', scale_units='xy', scale=2.2e4,
                    headlength=0, headwidth=0, headaxislength=0, zorder=2)
sc1 = ax[1].scatter(phigrid[0, predicted_system[0, ..., 0] > 0],
                    thetagrid[0, predicted_system[0, ..., 0] > 0],
                    s=0.05*np.abs(predicted_system[0, predicted_system[0, ..., 0] > 0, 0]),
                    color='k', marker='o', facecolors='none', zorder=1)
sc2 = ax[1].scatter(phigrid[0, predicted_system[0, ..., 0] < 0],
                    thetagrid[0, predicted_system[0, ..., 0] < 0],
                    s=0.05*np.abs(predicted_system[0, predicted_system[0, ..., 0] < 0, 0]),
                    color='k', marker='x', zorder=1)

quiv.set_clim(-1e4, 1e4)

ax[1].set_title('Predicted Current System', fontsize=title_fontsize)

ax[1].invert_yaxis()
ax[1].minorticks_on()
ax[1].set_xlabel('Longitude [deg]', fontsize=label_fontsize)
ax[1].set_ylabel('Co-latitude [deg]', fontsize=label_fontsize)
ax[1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[1].tick_params(which='minor', length=7, width=2)
ax[1].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))

lg1 = ax[1].legend(*sc1.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='upper left',
                    title='Upward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
lg2 = ax[1].legend(*sc2.legend_elements('sizes', num=4, func=(lambda x: 20*x)), loc='lower left',
                    title='Downward FAC', fontsize=tick_labelsize, title_fontsize=label_fontsize)
ax[1].add_artist(lg1)

cbar = fig.colorbar(quiv, ax=ax.ravel().tolist(), shrink=0.8)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

figs.update({f'{mag_pos.shape[0]}magpos_multidirection_vector': fig})

title_fontsize = 35
suptitle_fontsize = 40
suplabel_fontsize = 30
tick_labelsize = 25

fig, ax = plt.subplots(2, 4, figsize=[40, 20], constrained_layout=True, sharex=True, sharey=True)
im = ax[0, 0].imshow(actual_system[0, ..., 1],
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[0, 0].minorticks_on()
ax[0, 0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[0, 0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[0, 0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[0, 0].tick_params(which='minor', length=7, width=2)
ax[0, 0].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[0, 0].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[0, 0].set_title('Actual\nco-latitudinal currents', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[0, 0], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

im = ax[1, 0].imshow(predicted_system[0, ..., 1],
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[1, 0].minorticks_on()
ax[1, 0].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[1, 0].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[1, 0].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[1, 0].tick_params(which='minor', length=7, width=2)
ax[1, 0].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[1, 0].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[1, 0].set_title('Predicted\nco-latitudinal currents', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[1, 0], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

im = ax[0, 1].imshow(actual_system[0, ..., 2],
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[0, 1].minorticks_on()
ax[0, 1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[0, 1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[0, 1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[0, 1].tick_params(which='minor', length=7, width=2)
ax[0, 1].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[0, 1].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[0, 1].set_title('Actual\nlongitudinal currents', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[0, 1], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)


im = ax[1, 1].imshow(predicted_system[0, ..., 2],
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[1, 1].minorticks_on()
ax[1, 1].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[1, 1].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[1, 1].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[1, 1].tick_params(which='minor', length=7, width=2)
ax[1, 1].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[1, 1].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[1, 1].set_title('Predicted\nlongitudinal currents', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[1, 1], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

im = ax[0, 2].imshow(actual_system[0, ..., 0]*(actual_system[0, ..., 0] < 0),
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[0, 2].minorticks_on()
ax[0, 2].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[0, 2].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[0, 2].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[0, 2].tick_params(which='minor', length=7, width=2)
ax[0, 2].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[0, 2].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[0, 2].set_title('Actual\ndownward FACs', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[0, 2], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

im = ax[1, 2].imshow(predicted_system[0, ..., 0]*(predicted_system[0, ..., 0] < 0),
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[1, 2].minorticks_on()
ax[1, 2].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[1, 2].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[1, 2].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[1, 2].tick_params(which='minor', length=7, width=2)
ax[1, 2].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[1, 2].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[1, 2].set_title('Predicted\ndownward FACs', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[1, 2], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

im = ax[0, 3].imshow(actual_system[0, ..., 0]*(actual_system[0, ..., 0] >= 0),
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[0, 3].minorticks_on()
ax[0, 3].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[0, 3].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[0, 3].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[0, 3].tick_params(which='minor', length=7, width=2)
ax[0, 3].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[0, 3].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[0, 3].set_title('Actual\nupward FACs', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[0, 3], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

im = ax[1, 3].imshow(predicted_system[0, ..., 0]*(predicted_system[0, ..., 0] >= 0),
                        extent=[phigrid[0, 0, 0], phigrid[0, 0, -1], thetagrid[0, -1, 0], thetagrid[0, 0, 0]])

ax[1, 3].minorticks_on()
ax[1, 3].set_xticks(np.arange(phigrid[0, 0, 0], phigrid[0, 0, -1]+1, 3))
ax[1, 3].set_yticks(np.arange(thetagrid[0, 0, 0], thetagrid[0, -1, 0]+1, 2))
ax[1, 3].tick_params(which='major', length=10, width=3, labelsize=tick_labelsize)
ax[1, 3].tick_params(which='minor', length=7, width=2)
ax[1, 3].xaxis.set_minor_locator(AutoMinorLocator(6))
ax[1, 3].yaxis.set_minor_locator(AutoMinorLocator(4))

ax[1, 3].set_title('Predicted\nupward FACs', fontsize=title_fontsize)

cbar = fig.colorbar(im, ax=ax[1, 3], shrink=0.6)
cbar.ax.set_ylabel('I [A]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_labelsize)

fig.supxlabel('Longitude [deg]', fontsize=suplabel_fontsize)
fig.supylabel('Co-latitude [deg]', fontsize=suplabel_fontsize)
fig.suptitle('Actual vs. Predicted Current Values in Grid', fontsize=suptitle_fontsize)

figs.update({f'{mag_pos.shape[0]}magpos_multidirection_values': fig})

for key, fig in figs.items():
    fig.savefig(fname=f'Results/{key}.jpg', bbox_inches='tight')


# %%
