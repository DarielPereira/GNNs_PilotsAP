import numpy as np

from functionsSetup import *
from __Unused.functionsComputeExpectations import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

import seaborn as sns
from functionsUtils import grid_parameters
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink


##Setting Parameters
nbrOfSetups = 50                # number of Monte-Carlo setups
nbrOfRealizations = 4           # number of channel realizations per setup

L = 1                           # number of APs
N = 200                         # number of antennas per AP

K = 100                         # number of UEs

# Set the pilot reuse factor. Use T = 1 for single-cell
T = 3
tau_c = 200                    # length of coherence block
tau_p_ = 20                   # length of pilot sequences
# prelogFactor = 1-tau_p/tau_c    # uplink data transmission prelog factor

ASD_varphi = math.radians(5)  # Azimuth angle - Angular Standard Deviation in the local scattering model
ASD_theta = math.radians(15)   # Elevation angle - Angular Standard Deviation in the local scattering model

p = 100                         # total uplink transmit power per UE

# To save the simulation results
NMSE = np.zeros((K, nbrOfSetups))       # MMSE/all APs serving all the UEs

settings = {
   'clustering_mode': ['Kbeams', 'OPA', 'OPA_LU'],
   # 'clustering_mode': ['Kbeams'],
   'PA_mode': ['bf_NMSE'],
   'init_mode':['dissimilar_K'],
   'K': [20, 30, 40, 50, 60, 70, 80, 90, 100]
   # 'K': [30]
   }

results = {
'Kbeams':[0, 0, 0, 0, 0, 0, 0, 0, 0],
'OPA':[0, 0, 0, 0, 0, 0, 0, 0, 0],
'OPA_LU':[0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# iterate over pilot allocation modes
for setting in grid_parameters(settings):

    modes = {'clustering_mode': setting['clustering_mode'], 'PA_mode': setting['PA_mode'],
             'init_mode': setting['init_mode']}
    K = setting['K']
    print(f'K: {K}')
    SE_MMSEs = 0


    # iterate over the setups
    for iter in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(iter, nbrOfSetups))

        if setting['clustering_mode'] == 'OPA':
            tau_p = K
        else:
            tau_p = tau_p_

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, R, pilotIndex, clustering, D, D_small = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta,
                                                                   nbrOfRealizations, seed=iter, **modes)


        # # Compute NMSE for all the UEs
        # system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

        # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
        SE_MMSE = functionComputeSE_uplink(Hhat, H, D, D_small, B, C, tau_c, tau_p, T,
                                       nbrOfRealizations, N, K, L, p, R, pilotIndex)



        SE_MMSEs += float(sum(SE_MMSE).real)/nbrOfSetups

    results[modes['clustering_mode']][int(K / 10 - 2)] = SE_MMSEs


results_array = np.array(np.concatenate((np.matrix(results['Kbeams']), np.matrix(results['OPA']),
                                             np.matrix(results['OPA_LU'])), axis=0))

np.savez(f'./GRAPHs/VARIABLES_SAVED/SR_K_20_100_Iter_{nbrOfSetups}_tau_{tau_p_}_T_{T}',
                 results_array=results_array)


print('end')



