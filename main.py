import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as alg
import math
from functionsSetup import *
from functionsChannelEstimates import *
from functionsComputeSE_uplink import functionComputeSE_uplink
from functionsComputeExpectations import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink


##Setting Parameters
nbrOfSetups = 1                 # number of Monte-Carlo setups
nbrOfRealizations = 1           # number of channel realizations per setup

L = 150                        # number of APs
N = 4                           # number of antennas per AP

K = 100                          # number of UEs

tau_c = 5                    # length of coherence block
tau_p = 10                    # length of pilot sequences
prelogFactor = 1-tau_p/tau_c    # uplink data transmission prelog factor

ASD_varphi = math.radians(5)  # Azimuth angle - Angular Standard Deviation in the local scattering model
ASD_theta = math.radians(15)   # Elevation angle - Angular Standard Deviation in the local scattering model

p = 100                         # total uplink transmit power per UE

# To save the simulation results
NMSE = np.zeros((K, nbrOfSetups))       # MMSE/all APs serving all the UEs

# Pilot allocation modes ['DCC', 'Kmeans_basic_positions', 'Kmeans_basic_R']
pilot_alloc_modes= ['DCC', 'Kmeans_basic_positions', 'Kmeans_basic_R']

# iterate over pilot allocation modes
for pilot_alloc_mode in pilot_alloc_modes:
    # iterate over the setups
    for n in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(n, nbrOfSetups))

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, R, pilotIndex, D, D_small = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta,
                                                                   1, pilot_alloc_mode=pilot_alloc_mode)

        # Compute NMSE for all the UEs
        system_NMSE, UEs_NMSE, average_NMSE = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)


# # Plot Figure 5.8
# plt.figure()
# plt.plot(np.sort(SE_nopt_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b-', linewidth=2)
# plt.plot(np.sort(SE_LMMSE_original.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k-', linewidth=2)
# plt.plot(np.sort(SE_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'r-.', linewidth=2)
# plt.plot(np.sort(SE_Dist_MR_original.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=2)
# plt.plot(np.sort(SE_Dist_MR_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b--', linewidth=2)
# plt.xlabel('Spectral efficiency [bit/s/Hz]', fontsize=16)
# plt.ylabel('CDF', fontsize=16)
# plt.legend(['n-opt LSFD, LP-MMSE', 'L-MMSE (All)', 'LP-MMSE (DCC)', 'MR (All)', 'MR (DCC)'],
#            fontsize=16)
# plt.xlim([0, 12])
# plt.show()


print('end')



