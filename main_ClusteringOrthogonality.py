import numpy as np

from functionsSetup import *
from __Unused.functionsComputeExpectations import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

import seaborn as sns
from functionsUtils import grid_parameters


##Setting Parameters
nbrOfSetups = 100                # number of Monte-Carlo setups
nbrOfRealizations = 1           # number of channel realizations per setup

L = 1                           # number of APs
N = 200                         # number of antennas per AP

K = 100                         # number of UEs

tau_c = 5                    # length of coherence block
tau_p = 10                   # length of pilot sequences
prelogFactor = 1-tau_p/tau_c    # uplink data transmission prelog factor

ASD_varphi = math.radians(5)  # Azimuth angle - Angular Standard Deviation in the local scattering model
ASD_theta = math.radians(15)   # Elevation angle - Angular Standard Deviation in the local scattering model

p = 100                         # total uplink transmit power per UE

# To save the simulation results
NMSE = np.zeros((K, nbrOfSetups))       # MMSE/all APs serving all the UEs

settings = {
   'clustering_mode': ['Kmeans_basic_positions', 'Kbeams'],
   'PA_mode': ['none'],
   'init_mode':['dissimilar_K'],
   'K': [20, 30, 40, 50, 60, 70, 80, 90, 100]
   }

results = {
'Kmeans_basic_positions_in':[0, 0, 0, 0, 0, 0, 0, 0, 0],
'Kmeans_basic_positions_out':[0, 0, 0, 0, 0, 0, 0, 0, 0],
'Kbeams_in':[0, 0, 0, 0, 0, 0, 0, 0, 0],
'Kbeams_out':[0, 0, 0, 0, 0, 0, 0, 0, 0],

}

# iterate over pilot allocation modes
for setting in grid_parameters(settings):

    modes = {'clustering_mode': setting['clustering_mode'], 'PA_mode': setting['PA_mode'],
             'init_mode': setting['init_mode']}
    K = setting['K']

    intra_cell = 0
    outer_cell = 0

    # iterate over the setups
    for iter in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(iter, nbrOfSetups))

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, R, pilotIndex, clustering, D, D_small = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta,
                                                                   nbrOfRealizations, seed=iter, **modes)

        same_cluster, = np.where(clustering == clustering[0])
        other_cluster, = np.where(clustering != clustering[0])

        intra_cell += (np.trace(sum([R[:, :, 0, k]/(np.linalg.norm(R[:, :, 0, k])) for k in same_cluster])
                              @ R[:, :, 0, 0]/(np.linalg.norm(R[:, :, 0, 0])))/nbrOfSetups).real

        if len(other_cluster)>0:
            outer_cell += (np.trace(sum([R[:, :, 0, k]/(np.linalg.norm(R[:, :, 0, k])) for k in other_cluster])
                              @ R[:, :, 0, 0]/(np.linalg.norm(R[:, :, 0, 0])))/nbrOfSetups).real

    if setting['clustering_mode']=='Kmeans_basic_positions':
        results['Kmeans_basic_positions_in'][int(K / 10 - 2)] = intra_cell
        results['Kmeans_basic_positions_out'][int(K / 10 - 2)] = outer_cell
    elif setting['clustering_mode']=='Kbeams':
        results['Kbeams_in'][int(K / 10 - 2)] = intra_cell
        results['Kbeams_out'][int(K / 10 - 2)] = outer_cell

Kmeans_basic_positions = np.array(np.concatenate((np.matrix(results['Kmeans_basic_positions_in']),
                                         np.matrix(results['Kmeans_basic_positions_out'])), axis=0))
K_beams = np.array(np.concatenate((np.matrix(results['Kbeams_in']),
                                         np.matrix(results['Kbeams_out'])), axis=0))

np.savez(f'./GRAPHs/VARIABLES_SAVED/ClusteringOrthogonality_'+str(round(ASD_varphi, 2)),
         Kmeans_basic_positions=Kmeans_basic_positions, K_beams=K_beams)




