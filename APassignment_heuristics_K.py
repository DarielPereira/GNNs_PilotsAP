import math
import numpy as np
from nbclient.client import timestamp

from functionsSetup_heuristics import generateSetup
from functionsAllocation_heuristics import PilotAssignment, APassignment
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink
from functionsUtils import save_results
from datetime import datetime


##Setting Parameters
configuration = {
    'nbrOfSetups': 20,             # number of communication network setups
    'nbrOfRealizations': 3,      # number of channel realizations per sample
    'L': 49,                     # number of APs
    'N': 8,                       # number of antennas per AP
    'Ks': [20, 40, 60, 80, 100],                   # number of UEs
    'T': 49,                       # number of APs connected to each CPU
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 20,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'I': int(8),                  # number of APs to include in the graph
    'comb_mode': 'MR',           # combining method used to evaluate optimization
    'heuristic_modes': ['I_best', 'Graph', 'BPA', 'ALL']  # ['BPA', 'I_best', 'I_random', 'Graph', 'ALL']
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
Ks = configuration['Ks']
T = configuration['T']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
I = configuration['I']
comb_mode = configuration['comb_mode']
heuristic_modes = configuration['heuristic_modes']


results = {
    'Graph': np.zeros((len(Ks))),
    'BPA':  np.zeros((len(Ks))),
    'I_best':  np.zeros((len(Ks))),
    'ALL':  np.zeros((len(Ks)))
}

# run over all the values of K
for idx, K in enumerate(Ks):

    # run over the heuristic modes
    for heuristic_mode in heuristic_modes:
        print(f'Running heuristic mode: {heuristic_mode}')

        # Run over all the setups
        for setup_iter in range(nbrOfSetups):

            print(f'Generating setup {setup_iter + 1}/{nbrOfSetups} with {K} connected UEs......')

            # Generate one setup with UEs and APs at random locations
            gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
                generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=True, seed=setup_iter))

            # Compute AP and pilot assignment
            pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

            # Generate channel realizations with estimates and estimation error matrices
            Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

            D = APassignment(nbrOfRealizations, R, gainOverNoisedB, Hhat, H, B, C, p, tau_c, tau_p, L, N, K, I, pilotIndex,
                             mode = heuristic_mode, comb_mode = comb_mode)

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D,
                                                                           C, tau_c, tau_p,
                                                                           nbrOfRealizations, N,
                                                                           K, L, p)
            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mismatching')
                    SE = 0

            results[heuristic_mode][idx] += np.sum(SE)/nbrOfSetups

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f'./GRAPHs_heuristics/VARIABLES_SAVED/SE_K_Comb_'+comb_mode+f'_NbrSetups_{nbrOfSetups}_{L}_{N}_'+timestamp+'.pkl'
save_results(results, file_name)