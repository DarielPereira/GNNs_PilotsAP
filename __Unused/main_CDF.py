from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math
from functionsUtils import save_results
import numpy as np

from functionsAllocation import AP_PilotAssignment_UEsBLock, AP_Pilot_newUE
from functionsSetup import generateSetup, insertNewUE
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink



##Setting Parameters
configuration = {
    'nbrOfSetups': 10,             # number of communication network setups
    'nbrOfInitiallyConnectedlUEs':0,   # numnber of UEs already connected
    'nbrOfNewUEs': 200,            # number of UEs to insert
    'nbrOfRealizations': 5,      # number of channel realizations per sample
    'L': 225,                     # number of APs
    'N': 1,                       # number of antennas per AP
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 10,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'M': int(7),                  # number of APs to include in the graph
    'comb_mode': 'MR'           # combining method used to evaluate optimization
}

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']
nbrOfInitiallyConnectedlUEs = configuration['nbrOfInitiallyConnectedlUEs']
nbrOfNewUEs = configuration['nbrOfNewUEs']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
M = configuration['M']
comb_mode = configuration['comb_mode']

# TO STORE DATA
results = {
    'Optimal_SE': np.zeros((nbrOfNewUEs * nbrOfSetups)),
    'DCC_SE':  np.zeros((nbrOfNewUEs * nbrOfSetups)),
    'ALL_SE':  np.zeros((nbrOfNewUEs * nbrOfSetups)),
    'Optimal_nbrServingAPs': np.zeros((nbrOfNewUEs * nbrOfSetups)),
    'DCC_nbrServingAPs':  np.zeros((nbrOfNewUEs * nbrOfSetups)),
    'ALL_nbrServingAPs':  np.zeros((nbrOfNewUEs * nbrOfSetups))
    }

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    nbrOfConnectedlUEs = nbrOfInitiallyConnectedlUEs

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions = (
        generateSetup(L, nbrOfConnectedlUEs, N, cell_side, ASD_varphi, seed=setup_iter))

    # Compute AP and pilot assignment
    pilotIndex, D = AP_PilotAssignment_UEsBLock(R, gainOverNoisedB, tau_p, L, N, mode='DCC')

    # Run over the samples for each setup
    for newUE_idx in range(nbrOfNewUEs):
        print(f'Generating sample {newUE_idx+1} of setup {setup_iter+1}')

        gainOverNoisedB, distances, R, APpositions, UEpositions = (
            insertNewUE(L, N, cell_side, ASD_varphi, gainOverNoisedB, distances, R, APpositions,
                        UEpositions, seed=newUE_idx+1000))

        nbrOfConnectedlUEs += 1

        # Compute AP and pilot assignment
        pilotIndex, D = AP_Pilot_newUE(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, L, N,
                                       M, D, pilotIndex, comb_mode)

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, nbrOfConnectedlUEs, N, tau_p, pilotIndex, p)

    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
    SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, C, tau_c, tau_p,
                                       nbrOfRealizations, N, nbrOfConnectedlUEs, L, p)

    pilotIndex_DCC, D_DCC = AP_PilotAssignment_UEsBLock(R, gainOverNoisedB, tau_p, L, N, mode='DCC')

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, nbrOfConnectedlUEs, N, tau_p, pilotIndex_DCC, p)

    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
    SE_MMSE_DCC, SE_P_RZF_DCC, SE_MR_DCC, SE_P_MMSE_DCC = functionComputeSE_uplink(Hhat, H, D_DCC, C, tau_c, tau_p,
                                   nbrOfRealizations, N, nbrOfConnectedlUEs, L, p)

    D_ALL = np.ones((L, nbrOfConnectedlUEs))

    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
    SE_MMSE_ALL, SE_P_RZF_ALL, SE_MR_ALL, SE_P_MMSE_ALL = functionComputeSE_uplink(Hhat, H, D_ALL, C, tau_c, tau_p,
                                                                    nbrOfRealizations, N, nbrOfConnectedlUEs, L, p)


    print('Ave. number of serving APs Optimal: ', np.mean(np.sum(D, axis=0)))
    print('Ave. number of serving APs DCC: ', np.mean(np.sum(D_DCC, axis=0)))
    print('Ave. number of serving APs ALL: ', np.mean(np.sum(D_ALL, axis=0)))

    match comb_mode:
        case 'MR':
            results['Optimal_SE'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = SE_MR[:].flatten()
            results['DCC_SE'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = SE_MR_DCC[:].flatten()
            results['ALL_SE'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = SE_MR_ALL[:].flatten()
            results['Optimal_nbrServingAPs'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = (
                np.sum(D, axis=0))
            results['DCC_nbrServingAPs'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = (
                np.sum(D_DCC, axis=0))
            results['ALL_nbrServingAPs'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = (
                np.sum(D_ALL, axis=0))

            print('Sum SE Optimal: ', np.sum(SE_MR))
            print('Sum SE DCC: ', np.sum(SE_MR_DCC))
            print('Sum SE ALL: ', np.sum(SE_MR_ALL))

        case 'MMSE':
            results['Optimal_SE'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = SE_MMSE[:].flatten()
            results['DCC_SE'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = SE_MMSE_DCC[:].flatten()
            results['ALL_SE'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = SE_MMSE_ALL[:].flatten()
            results['Optimal_nbrServingAPs'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = (
                np.sum(D, axis=0))
            results['DCC_nbrServingAPs'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = (
                np.sum(D_DCC, axis=0))
            results['ALL_nbrServingAPs'][setup_iter * nbrOfNewUEs:(setup_iter + 1) * nbrOfNewUEs] = (
                np.sum(D_ALL, axis=0))

            print('Sum SE Optimal: ', np.sum(SE_MMSE))
            print('Sum SE DCC: ', np.sum(SE_MMSE_DCC))
            print('Sum SE ALL: ', np.sum(SE_MMSE_ALL))

file_name = f'./GRAPHs/VARIABLES_SAVED/SE_CDF_Comb_'+comb_mode+f'_NbrSetups_{nbrOfSetups}.pkl'
save_results(results, file_name)
