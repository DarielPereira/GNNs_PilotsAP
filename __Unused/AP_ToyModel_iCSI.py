import math
import numpy as np
from functionsSetup import generateSetup, insertNewUE
from functionsAllocation import AP_PilotAssignment_UEsBLock, AP_Pilot_newUE, toyModel_AP_Pilot_newUE
from functionsUtils import save_results
from functionsUtils import drawingSetup





##Setting Parameters
configuration = {
    'nbrOfSetups': 1,             # number of communication network setups
    'nbrOfInitiallyConnectedlUEs': 0,   # numnber of UEs already connected
    'nbrOfNewUEs': 6,            # number of UEs to insert
    'nbrOfRealizations': 50,      # number of channel realizations per sample
    'L': 9,                     # number of APs
    'N': 1,                       # number of antennas per AP
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 3,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 100,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'M': int(4),                  # number of APs to include in the graph
    'comb_mode': 'MR',           # combining method used to evaluate optimization
    'update_mode': 'newUE',  # update mode used to evaluate optimization
    'csi_mode': 'imperfect'  # csi mode used to evaluate optimization
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
update_mode = configuration['update_mode']
csi_mode = configuration['csi_mode']


# to store the SE values
SEs_perSetup = np.zeros((nbrOfNewUEs, nbrOfNewUEs))

SEs = np.zeros((nbrOfNewUEs, nbrOfNewUEs))

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    # set the initial number of connected UEs for each setup
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

        # Select update modes = ['newUE_all', 'newUE_local', 'allUEs']
        # Compute AP and pilot assignment
        pilotIndex, D, best_sum_SE, best_SE = toyModel_AP_Pilot_newUE(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c,
                                nbrOfConnectedlUEs, L, N, M, D, pilotIndex,
                                comb_mode, update_mode = update_mode, csi_mode = csi_mode)

        # printing
        print(f'Best sum SE: {best_sum_SE}')
        print(f'Best SE: {best_SE}')
        print('pilotIndex:', pilotIndex)
        print('D:', D)

        SEs_perSetup[newUE_idx, 0:newUE_idx+1] = best_SE.flatten()

    SEs += SEs_perSetup/nbrOfSetups

    # setup map
    drawingSetup(UEpositions, APpositions, np.arange(0, nbrOfConnectedlUEs), title="Setup Map", squarelength=cell_side)

file_name = (f'./ToyModelsData/ToyModelsSE_{update_mode}_CSI_{csi_mode}_NbrSetups_{nbrOfSetups}'
             f'_NbrUEs_{nbrOfNewUEs}.pkl')
save_results(SEs, file_name)