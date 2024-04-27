import numpy as np

from functionsSetup import *
from __Unused.functionsComputeExpectations import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

import seaborn as sns
from functionsUtils import grid_parameters


##Setting Parameters
nbrOfSetups = 10                # number of Monte-Carlo setups
nbrOfRealizations = 1           # number of channel realizations per setup

L = 1                           # number of APs
N = 200                         # number of antennas per AP

K = 100                         # number of UEs

tau_c = 5                    # length of coherence block
tau_p = 10                   # length of pilot sequences
prelogFactor = 1-tau_p/tau_c    # uplink data transmission prelog factor

ASD_varphi = math.radians(5)  # Azimuth angle - Angular Standard Deviation in the local scattering model
ASD_theta = math.radians(15)   # Elevation angle - Angular Standard Deviation in the local scattering model

p = 100                         # total uplink transmit power per UE

# To save the simulation results
NMSE = np.zeros((K, nbrOfSetups))       # MMSE/all APs serving all the UEs

# Clustering modes ['Kbeams', 'DCC', 'Kmeans_basic_positions', 'Kmeans_basic_R', 'SGPS']
# clustering_modes = ['Kbeams']
# In-clusters pilot allocation modes ['basic', 'worst_first', 'best_first', 'bf_NMSE', 'wf_NMSE']
# PA_mode = 'bf_NMSE'

classic_parameters = {
    # Select from ['DCC', 'SGPS', 'random']
    'PA_mode': ['DCC', 'SGPS']
}
Kmeans_parameters = {
    # Select from ['Kmeans_basic_positions']
    'clustering_mode': ['Kmeans_basic_positions'],
    # Select from ['basic', 'worst_first', 'best_first', 'bf_NMSE', 'wf_NMSE']
    'PA_mode': ['best_first', 'worst_first']
}
Kbeams_parameters = {
    # Select from ['Kbeams']
    'clustering_mode': ['Kbeams'],
    # Select from ['basic', 'worst_first', 'best_first', 'bf_NMSE', 'wf_NMSE']
    'PA_mode': ['bf_NMSE'],
    # select from ['basic_angle_spread', 'dissimilar_K']
    'init_mode': ['basic_angle_spread', 'dissimilar_K']
}

# iterate over pilot allocation modes
for setting in grid_parameters(classic_parameters):
    system_NMSEs = []
    UEs_NMSEs = []
    ordered_UEs_NMSEs = []

    # iterate over the setups
    for iter in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(iter, nbrOfSetups))

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, R, pilotIndex, clustering, D, D_small = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta,
                                                                   nbrOfRealizations, seed=iter, **setting)

        # Compute NMSE for all the UEs
        system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

        system_NMSEs.append(system_NMSE)
        UEs_NMSEs.append(UEs_NMSE)
        ordered_UEs_NMSEs.append(np.sort(UEs_NMSE))

    if len(setting) == 1:
        np.savez(f'./VARIABLES_SAVED/{setting['PA_mode']}',
                 system_NMSEs=system_NMSEs, UEs_NMSEs=UEs_NMSEs, ordered_UEs_NMSEs=ordered_UEs_NMSEs)

    elif len(setting) == 2:
        np.savez(f'./VARIABLES_SAVED/{setting['clustering_mode']}__{setting['PA_mode']}',
                 system_NMSEs=system_NMSEs, UEs_NMSEs=UEs_NMSEs, ordered_UEs_NMSEs=ordered_UEs_NMSEs)

    elif len(setting) == 3:
        np.savez(f'./VARIABLES_SAVED/{setting['clustering_mode']}__{setting['PA_mode']}__{setting['init_mode']}',
                 system_NMSEs=system_NMSEs, UEs_NMSEs=UEs_NMSEs, ordered_UEs_NMSEs=ordered_UEs_NMSEs)


print('end')



