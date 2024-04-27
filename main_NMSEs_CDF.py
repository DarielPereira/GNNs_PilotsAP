import numpy as np

from functionsSetup import *
from __Unused.functionsComputeExpectations import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

import seaborn as sns
from functionsUtils import grid_parameters


##Setting Parameters
nbrOfSetups = 30                # number of Monte-Carlo setups
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

settings = {
   'clustering_mode': ['Kmeans_basic_positions', 'Kbeams', 'DCC', 'SGPS'],
   'PA_mode': ['bf_NMSE'],
   'init_mode':['dissimilar_K'],
   }

results = {
'Kmeans_basic_positions':np.zeros((nbrOfSetups*K)),
'DCC':np.zeros((nbrOfSetups*K)),
'Kbeams':np.zeros((nbrOfSetups*K)),
'SGPS':np.zeros((nbrOfSetups*K)),

}

# iterate over pilot allocation modes
for setting in grid_parameters(settings):

    modes = {'clustering_mode': setting['clustering_mode'], 'PA_mode': setting['PA_mode'],
             'init_mode': setting['init_mode']}

    system_NMSEs = 0

    # iterate over the setups
    for iter in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(iter, nbrOfSetups))

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, R, pilotIndex, clustering, D, D_small = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta,
                                                                   nbrOfRealizations, seed=iter, **modes)

        # Compute NMSE for all the UEs
        system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

        system_NMSEs += system_NMSE/nbrOfSetups

        results[modes['clustering_mode']][iter*K:(iter+1)*K] = UEs_NMSE



results_array = np.array(np.concatenate((np.matrix(results['Kmeans_basic_positions']),
                             np.matrix(results['DCC']), np.matrix(results['Kbeams']),
                                             np.matrix(results['SGPS'])), axis=0))

np.savez(f'./GRAPHs/VARIABLES_SAVED/NMSEs_CDF_Iter_{nbrOfSetups}',
                 results_array=results_array)


print('end')



