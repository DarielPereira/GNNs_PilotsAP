import numpy as np

from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

from functionsClustering import cfMIMO_clustering
from functionsAllocation import pilotAssignment
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink


##Setting Parameters
configuration = {
    'nbrOfSetups': 1,             # number of Monte-Carlo setups
    'nbrOfRealizations': 1,       # number of channel realizations per setup
    'L': 100,                       # number of APs
    'N': 4,                     # number of antennas per AP
    'K': 100,                     # number of UEs
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 10,                  # length of the pilot sequences
    'T': 1,                       # pilot reuse factor
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'ASD_theta': math.radians(15),          # Elevation angle - Angular Standard Deviation in the local scattering model
    'cl_mode': 'no_clustering',             # clustering mode from
    # ['Kmeans_locations', 'Kfootprints', 'no_clustering']
    'pa_mode': 'DCC'                        # pilot allocation mode from
    # ['random', 'balanced_random', 'DCPA', 'DCC', 'basic_iC', 'bf_iC', 'bfNMSE_iC', 'bf_bAPs_iC']
}

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
K = configuration['K']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
T = configuration['T']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
ASD_theta = configuration['ASD_theta']


# iterate over the setups
for iter in range(nbrOfSetups):
    print("Setup iteration {} of {}".format(iter, nbrOfSetups))

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions = (
        generateSetup(L, K, N, cell_side, ASD_varphi, seed=2))

    # clustering modes ['Kmeans_locations', 'Kfootprints', 'no_clustering']
    UE_clustering \
        = cfMIMO_clustering(gainOverNoisedB, R, tau_p, APpositions, UEpositions, mode=configuration['cl_mode'])

    # pilot assignment modes ['random', 'balanced_random', 'DCPA', 'DCC', 'basic_iC', 'bf_iC', 'bfNMSE_iC', 'bf_bAPs_iC']
    pilotIndex, D = pilotAssignment(UE_clustering, R, gainOverNoisedB, K, tau_p, L, N, mode=configuration['pa_mode'])

    # Compute NMSE for all the UEs
    system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot \
        = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

    print('System NMSE: {}'.format(system_NMSE))

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
    SE_MMSE = functionComputeSE_uplink(Hhat, H, D, C, tau_c, tau_p, T,
                                       nbrOfRealizations, N, K, L, p, R, pilotIndex)

    print('Sum-rate: {}'.format(sum(SE_MMSE)))

print('end')



