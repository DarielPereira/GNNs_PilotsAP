from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math


from functionsAllocation import AP_PilotAssignment_UEsBLock, AP_PilotAssignment_newUE
from functionsSetup import generateSetup, insertNewUE
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink


##Setting Parameters
configuration = {
    'nbrOfSetups': 1,             # number of communication network setups
    'nbrOfConnectedlUEs':100,   # numnber of UEs already connected
    'nbrOfNewUEs': 50,            # number of UEs to insert
    'nbrOfRealizations': 5,      # number of channel realizations per sample
    'L': 225,                     # number of APs
    'N': 1,                       # number of antennas per AP
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 3,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'nbrOfCloseAPs': 10           # number of APs to include in the graph
}

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']
nbrOfConnectedlUEs = configuration['nbrOfConnectedlUEs']
nbrOfNewUEs = configuration['nbrOfNewUEs']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']

# TO STORE DATA
# Matrix with the correlation matrices

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions = (
        generateSetup(L, nbrOfConnectedlUEs, N, cell_side, ASD_varphi, seed=2))

    # Compute AP and pilot assignment
    pilotIndex, D = AP_PilotAssignment_UEsBLock(R, gainOverNoisedB, tau_p, L, N, mode='ALL')


    # Run over the samples for each setup
    for newUE_idx in range(nbrOfNewUEs):
        print(f'Generating sample {newUE_idx+1} of setup {setup_iter+1}')

        gainOverNoisedB, distances, R, APpositions, UEpositions = (
            insertNewUE(L, N, cell_side, ASD_varphi, gainOverNoisedB, distances, R, APpositions,
                        UEpositions, seed=newUE_idx+1000))

        nbrOfConnectedlUEs += 1

        # Compute AP and pilot assignment
        pilotIndex, D = AP_PilotAssignment_newUE(R, gainOverNoisedB, tau_p,  L, N, pilotIndex, mode='ALL')

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, nbrOfConnectedlUEs, N, tau_p, pilotIndex, p)

        # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
        SE_MMSE = functionComputeSE_uplink(Hhat, H, D, C, tau_c, tau_p,
                                           nbrOfRealizations, N, nbrOfConnectedlUEs, L, p, R, pilotIndex)

        print('Sum-rate: {}'.format(sum(SE_MMSE)))





