from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math
from functionsUtils import save_results
import numpy as np
import torch as th

from functionsAllocation import AP_PilotAssignment_UEsBLock, AP_Pilot_newUE, AP_Pilot_GeneratingSamples
from functionsSetup import generateSetup, insertNewUE
from functionsGraphHandling import SampleBuffer, get_AP2UE_edges, get_Pilot2UE_edges, get_oneHot_bestPilot
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink



##Setting Parameters
configuration = {
    'nbrOfSetups': 10,             # number of communication network setups
    'nbrOfInitiallyConnectedlUEs': 0,   # numnber of UEs already connected
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

# Create the sample storage buffer
sampleBuffer = SampleBuffer(batch_size=10)

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

        # Compute AP and pilot assignment
        pilotIndex, D, D_sample, R_sample, pilotIndex_sample, best_pilot_sample, best_APassignment_sample \
            = AP_Pilot_GeneratingSamples(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, L, N,
                                       M, D, pilotIndex, comb_mode)

        # Convert the data to graph samples
        # Ap2UE edges
        AP2UE_edges = get_AP2UE_edges(D_sample)
        # Pilot2UE edges
        Pilot2UE_edges = get_Pilot2UE_edges(pilotIndex_sample)
        # One-hot encoding torch of the best pilot
        oneHot_bestPilot = get_oneHot_bestPilot(best_pilot_sample, tau_p)
        # Best pilot assignment torch
        best_APassignment = th.tensor(best_APassignment_sample)
        # R matrix torch for the previously connected UEs
        R_features = th.tensor(R_sample[:, :, :, :-1])
        # R matrix torch for the new UE
        R_newUE = th.tensor(R_sample[:, :, :, -1])

        # Store the tuple samples in the buffer
        sampleBuffer.add((AP2UE_edges, Pilot2UE_edges, oneHot_bestPilot, best_APassignment, R_features, R_newUE))



file_name = (
        f'./TRAININGDATA/SE_Comb_'+comb_mode+f'_L_{L}_N_{N}_M_{M}_taup_{tau_p}_NbrSamp_{nbrOfSetups*nbrOfNewUEs}.pkl')

sampleBuffer.save(file_name)


