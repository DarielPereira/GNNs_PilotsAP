import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as alg
import math
from functionsSetup import *
from functionsChannelEstimates import *

##Setting Parameters
nbrOfSetups = 2                 # number of Monte-Carlo setups
nbrOfRealizations = 3           # number of channel realizations per setup

L = 5                        # number of APs
N = 4                           # number of antennas per AP

K = 3                          # number of UEs

tau_c = 200                     # length of coherence block
tau_p = 10                      # lenghth of pilot sequences
prelogFactor = 1-tau_p/tau_c    # uplink data transmission prelog factor

ASD_varphi = math.radians(15)  # Azimuth angle - Angular Standard Deviation in the local scattering model
ASD_theta = math.radians(15)   # Elevation angle - Angular Standard Deviation in the local scattering model

p = 100                         # total uplink transmit power per UE

# To save the simulation results
SE_MMSE_original = np.zeros((K, nbrOfSetups))       # MMSE/all APs serving all the UEs
SE_MMSE_DCC = np.zeros((K, nbrOfSetups))            # MMSE/DCC
SE_PMMSE_DCC = np.zeros((K, nbrOfSetups))            # P-MMSE/DCC
SE_PRZF_DCC = np.zeros((K, nbrOfSetups))             # P-RZF/DCC
SE_MR_DCC = np.zeros((K, nbrOfSetups))               # C-MR/DCC

SE_opt_LMMSE_original = np.zeros((K, nbrOfSetups))   # opt LSFD, L-MMSE/all APs serving all the UEs
SE_opt_LMMSE_DCC = np.zeros((K, nbrOfSetups))        # opt LSFD, L-MMSE/DCC
SE_nopt_LPMMSE_DCC = np.zeros((K, nbrOfSetups))      # n-opt LSFD, LP-MMSE/DCC
SE_nopt_MR_DCC = np.zeros((K, nbrOfSetups))          # n-opt LSFD, MR/DCC

SE_LMMSE_original = np.zeros((K, nbrOfSetups))       # L-MMSE/all APs serving all the UEs
SE_LPMMSE_DCC = np.zeros((K, nbrOfSetups))           # LP-MMSE/DCC
SE_Dist_MR_original = np.zeros((K, nbrOfSetups))     # D-MR/all APs serving all the UEs
SE_Dist_MR_DCC = np.zeros((K, nbrOfSetups))          # D-MR/DCC

#iterate over the setups
for n in range(nbrOfSetups):
    print("Setup iteration {} of {}".format(n, nbrOfSetups))

    # Generate on setup with UEs and APs at random locations
    gainOverNoisedB, R, pilotIndex, D, D_small = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta, nbrOfSetups=1, seed=2)

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)


    # SE for cell-free massive MIMO
    # AP-UE allocation matrix when all the APs serve all the UEs
    D_all = np.ones((L, K))

    #Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
    functionComputeSE_uplink(Hhat, H, D_all, D_small, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex)






    print('end')



