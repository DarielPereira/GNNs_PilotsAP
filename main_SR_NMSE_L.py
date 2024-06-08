import numpy as np

from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

from functionsClustering import cfMIMO_clustering
from functionsPilotAlloc import pilotAssignment
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink
from functionsUtils import save_results, load_results


##Setting Parameters
configuration = {
    'nbrOfSetups': 30,             # number of Monte-Carlo setups
    'nbrOfRealizations': 3,       # number of channel realizations per setup
    'K': 100,                     # number of UEs
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 10,                  # length of the pilot sequences
    'T': 1,                       # pilot reuse factor
    'p': 100,                     # uplink transmit power per UE in mW
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'ASD_theta': math.radians(15),          # Elevation angle - Angular Standard Deviation in the local scattering model
}

algorithms = {
    'Kmeans': ['Kmeans_locations', 'bf_bAPs_iC'],
    'Kfootprints': ['Kfootprints', 'bf_bAPs_iC'],
    'DCC': ['no_clustering', 'DCC'],
    'DCPA': ['no_clustering', 'DCPA'],
    'balanced_random': ['no_clustering', 'balanced_random'],
    'random': ['no_clustering', 'random'],
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
K = configuration['K']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
T = configuration['T']
p = configuration['p']
ASD_varphi = configuration['ASD_varphi']
ASD_theta = configuration['ASD_theta']

setups = [(4, 100), (9, 36), (16, 25), (36, 16), (64, 9), (100, 4), (400, 1)]

results = {
    'Kmeans_locations': {'NMSEs': np.zeros((len(setups))), 'sum_rates': np.zeros((len(setups)))},
    'Kfootprints': {'NMSEs': np.zeros((len(setups))), 'sum_rates': np.zeros((len(setups)))},
    'DCC': {'NMSEs': np.zeros((len(setups))), 'sum_rates': np.zeros((len(setups)))},
    'DCPA': {'NMSEs': np.zeros((len(setups))), 'sum_rates': np.zeros((len(setups)))},
    'balanced_random': {'NMSEs': np.zeros((len(setups))), 'sum_rates': np.zeros((len(setups)))},
    'random': {'NMSEs': np.zeros((len(setups))), 'sum_rates': np.zeros((len(setups)))},
}

for idx, setup in enumerate(setups):
    L = setup[0]
    N = setup[1]

    for algorithm in algorithms:
        cl_mode = algorithms[algorithm][0]
        pa_mode = algorithms[algorithm][1]

        sum_rates = 0
        NMSEs = 0

        print(f'number of APs L: {L}')
        print(f'number of antennas N: {N}')
        print('Clustering mode: ' + cl_mode)
        print('Pilot allocation mode: ' + pa_mode)

        # iterate over the setups
        for iter in range(nbrOfSetups):
            print("Setup iteration {} of {}".format(iter+1, nbrOfSetups))

            # Generate one setup with UEs and APs at random locations
            gainOverNoisedB, distances, R, APpositions, UEpositions = (
                generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta, nbrOfRealizations, seed=iter))

            UE_clustering \
                = cfMIMO_clustering(gainOverNoisedB, R, tau_p, APpositions, UEpositions, mode=cl_mode)

            pilotIndex, D = pilotAssignment(UE_clustering, R, gainOverNoisedB, K, tau_p, L, N, mode=pa_mode)

            # Compute NMSE for all the UEs
            system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot \
                = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

            NMSEs += system_NMSE/nbrOfSetups
            print('System NMSE: {}'.format(system_NMSE))

            # Generate channel realizations with estimates and estimation error matrices
            Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE = functionComputeSE_uplink(Hhat, H, D, C, tau_c, tau_p, T,
                                               nbrOfRealizations, N, K, L, p, R, pilotIndex)

            print('Sum-rate: {}'.format(sum(SE_MMSE)[0]))
            sum_rates += sum(SE_MMSE)[0]/nbrOfSetups

        if cl_mode in results:
            results[cl_mode]['NMSEs'][idx] = NMSEs
            results[cl_mode]['sum_rates'][idx] = sum_rates
        elif pa_mode in results:
            results[pa_mode]['NMSEs'][idx] = NMSEs
            results[pa_mode]['sum_rates'][idx] = sum_rates

file_name = f'./GRAPHs/VARIABLES_SAVED/SR_NMSE_L_K_{K}_NbrSetps_{nbrOfSetups}_1_400.pkl'
save_results(results, file_name)