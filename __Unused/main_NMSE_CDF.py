from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

from functionsClustering import cfMIMO_clustering
from functionsPilotAlloc import pilotAssignment
from __Unused.functionsChannelEstimates import channelEstimates
from __Unused.functionsComputeSE_uplink import functionComputeSE_uplink
from functionsUtils import save_results

##Setting Parameters
configuration = {
    'nbrOfSetups': 50,             # number of Monte-Carlo setups
    'nbrOfRealizations': 1,       # number of channel realizations per setup
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

setups = [(4, 100), (9, 36), (100, 4), (400, 1)]

results = {
    'Kmeans_locations': np.zeros((len(setups), K * nbrOfSetups)),
    'Kfootprints': np.zeros((len(setups), K * nbrOfSetups)),
    'DCC': np.zeros((len(setups), K * nbrOfSetups)),
    'DCPA': np.zeros((len(setups), K * nbrOfSetups)),
    'balanced_random': np.zeros((len(setups), K * nbrOfSetups)),
    'random': np.zeros((len(setups), K * nbrOfSetups)),
}

for idx, setup in enumerate(setups):
    L = setup[0]
    N = setup[1]

    for algorithm in algorithms:
        cl_mode = algorithms[algorithm][0]
        pa_mode = algorithms[algorithm][1]

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

            print('System NMSE: {}'.format(system_NMSE))

            if cl_mode in results:
                results[cl_mode][idx, iter * K:(iter+1) * K] = UEs_NMSE[:]
            elif pa_mode in results:
                results[pa_mode][idx, iter * K:(iter+1) * K] = UEs_NMSE[:]

file_name = f'./GRAPHs/VARIABLES_SAVED/NMSE_CDF_K_{K}_NbrSetps_{nbrOfSetups}.pkl'
save_results(results, file_name)