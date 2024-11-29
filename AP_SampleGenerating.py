import math
from functionsUtils import save_results
import numpy as np
import torch as th
import random

from functionsAllocation import PilotAssignment, AP_GeneratingSamples
from functionsSetup import generateSetup
from functionsGraphHandling import SampleBuffer, get_AP2UE_edges, get_Pilot2UE_edges, get_oneHot_bestPilot
from functionsChannelEstimates import channelEstimates



##Setting Parameters
configuration = {
    'nbrOfSetups': 1,             # number of communication network setups
    'nbrOfConnectedUEs_range': [1, 150],            # number of UEs to insert
    'nbrOfRealizations': 5,      # number of channel realizations per sample
    'L': 225,                     # number of APs
    'N': 1,                       # number of antennas per AP
    'tau_c': 400,                 # length of the coherence block
    'tau_p': 150,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'M': int(7),                  # number of potential APs to include in the graph
    'I': int(3),                  # number of relevant UEs to include in the graph
    'comb_mode': 'MR',           # combining method used to evaluate optimization
    'potentialAPs_mode': 'base', # mode used to select the potential APs
    'relevantUEs_mode': 'base'   # mode used to select the relevant UEs
    }

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']
nbrOfConnectedUEs_range = configuration['nbrOfConnectedUEs_range']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
M = configuration['M']
I = configuration['I']
comb_mode = configuration['comb_mode']
potentialAPs_mode = configuration['potentialAPs_mode']
relevantUEs_mode = configuration['relevantUEs_mode']

# Create the sample storage buffer
sampleBuffer = SampleBuffer(batch_size=10)

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    # sample the number of connected UEs from a uniform distribution in the specified range (nbrOfConnectedUEs_range)
    # random.seed(setup_iter+nbrOfSetups)
    K = random.randint(nbrOfConnectedUEs_range[0], nbrOfConnectedUEs_range[1])
    # K = 2

    print(f'Generating setup {setup_iter+1}/{nbrOfSetups} with {K} connected UEs......')

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions = (
        generateSetup(L, K, N, cell_side, ASD_varphi, seed=setup_iter+nbrOfSetups))

    # Compute AP and pilot assignment
    pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

    sampleBuffer = AP_GeneratingSamples(sampleBuffer, p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, Hhat, H,
                                        B, C, L, K, N, M, I,
                   comb_mode, potentialAPs_mode, relevantUEs_mode)

file_name = (
f'./AP_TRAININGDATA/newData/SE_Comb_'
+comb_mode+f'_L_{L}_N_{N}_M_{M}_I_{I}_taup_{tau_p}_NbrSamp_{len(sampleBuffer.storage)}.pkl')

sampleBuffer.save(file_name)


