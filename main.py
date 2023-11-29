import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as alg
import math
from functionsSetup import *
from functionsChannelEstimates import *
from functionsComputeSE_uplink import *
from functionsComputeExpectations import *


##Setting Parameters
nbrOfSetups = 1                 # number of Monte-Carlo setups
nbrOfRealizations = 50           # number of channel realizations per setup

L = 100                        # number of APs
N = 4                           # number of antennas per AP

K = 40                          # number of UEs

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

SE_Genie_PMMSE_DCC = np.zeros((K,nbrOfSetups))      # P-MMSE (Genie-aided)
SE_UatF_PMMSE_DCC = np.zeros((K,nbrOfSetups))       # P-MMSE (YatF)

SE_opt_LMMSE_original = np.zeros((K, nbrOfSetups))   # opt LSFD, L-MMSE/all APs serving all the UEs
SE_opt_LMMSE_DCC = np.zeros((K, nbrOfSetups))        # opt LSFD, L-MMSE/DCC
SE_nopt_LPMMSE_DCC = np.zeros((K, nbrOfSetups))      # n-opt LSFD, LP-MMSE/DCC
SE_nopt_MR_DCC = np.zeros((K, nbrOfSetups))          # n-opt LSFD, MR/DCC

SE_Genie_nopt_LPMMSE_DCC = np.zeros((K,nbrOfSetups))    # n-opt LSFD, LP-MMSE (Genie-aided)
SE_Genie_nopt_MR_DCC = np.zeros((K,nbrOfSetups))        # n-opt LSFD, MR (Genie-aided)

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
    SE_MMSE_all, SE_P_MMSE_all, SE_P_RZF_all, SE_MR_cent_all, SE_opt_L_MMSE_all, SE_nopt_LP_MMSE_all, \
    SE_nopt_MR_all, SE_L_MMSE_all, SE_LP_MMSE_all, SE_MR_dist_all, Gen_SE_P_MMSE_all, Gen_SE_P_RZF_all, \
    Gen_SE_LP_MMSE_all, Gen_SE_MR_dist_all, SE_small_MMSE_all, Gen_SE_small_MMSE_all \
        = functionComputeSE_uplink(Hhat, H, D_all, D_small, B, C, tau_c, tau_p,
                                   nbrOfRealizations, N, K, L, p, R, pilotIndex)

    # Save SE values
    SE_MMSE_original[:, n] = SE_MMSE_all.flatten()
    SE_opt_LMMSE_original[:, n] = SE_opt_L_MMSE_all.flatten()
    SE_LMMSE_original[:, n] = SE_L_MMSE_all.flatten()
    SE_Dist_MR_original[:, n] = SE_MR_dist_all.flatten()

    # Compute SE for centralized and distributed ulpink operations for DCC
    SE_MMSE, SE_P_MMSE, SE_P_RZF, SE_MR_cent, SE_opt_L_MMSE, SE_nopt_LP_MMSE, \
    SE_nopt_MR, SE_L_MMSE, SE_LP_MMSE, SE_MR_dist, Gen_SE_P_MMSE, Gen_SE_P_RZF, \
    Gen_SE_LP_MMSE, Gen_SE_MR_dist, SE_small_MMSE, Gen_SE_small_MMSE \
        = functionComputeSE_uplink(Hhat, H, D, D_small, B, C, tau_c, tau_p,
                                   nbrOfRealizations, N, K, L, p, R, pilotIndex)

    SE_MMSE_DCC[:, n] =  SE_MMSE.flatten()
    SE_PMMSE_DCC[:, n] = SE_P_MMSE.flatten()
    SE_PRZF_DCC[:, n] = SE_P_RZF.flatten()
    SE_MR_DCC[:, n] =  SE_MR_cent.flatten()
    SE_opt_LMMSE_DCC[:, n] =  SE_opt_L_MMSE.flatten()
    SE_nopt_LPMMSE_DCC[:, n] =  SE_nopt_LP_MMSE.flatten()
    SE_nopt_MR_DCC[:, n] =  SE_nopt_MR.flatten()
    SE_Genie_PMMSE_DCC[:, n] = Gen_SE_P_MMSE.flatten()
    SE_Genie_nopt_LPMMSE_DCC[:, n] = Gen_SE_LP_MMSE.flatten()
    SE_Genie_nopt_MR_DCC[:, n] = Gen_SE_MR_dist.flatten()
    SE_LPMMSE_DCC[:, n]= SE_LP_MMSE.flatten()
    SE_Dist_MR_DCC[:, n]= SE_MR_dist.flatten()

    # Obtain full power vector
    p_full = p * np.ones(K)

    # Obtain the expectations for the computation of the UatF bound in Theorem 5.2
    signal_P_MMSE, signal2_P_MMSE, scaling_P_MMSE, signal_P_RZF, signal2_P_RZF, \
    scaling_P_RZF, signal_LP_MMSE, signal2_LP_MMSE, scaling_LP_MMSE \
        = ComputeExpectations(Hhat, H, D, C, nbrOfRealizations, N, K, L, p_full)

    # Prepare to store arrays for the terms in 5.9
    b_P_MMSE = np.zeros(K)
    C_P_MMSE = signal2_P_MMSE

    for k in range(K):
        # Compute the square root of the numerator term without p_k in 5.9
        b_P_MMSE[k] = np.abs(signal_P_MMSE[k, k])

        # Compute the denominator term without power terms p_i and the noise in 5.9
        C_P_MMSE[k, k] = C_P_MMSE[k, k] - np.abs(signal_P_MMSE[k, k])**2

    # Compute the numerator term without p_k in 5.9 for all UEs
    bk_P_MMSE = b_P_MMSE**2

    # Compute the interference term without p_i's in (5.9) for all UEs
    ck_P_MMSE = C_P_MMSE.real

    # Compute the effective noise variance in (5.9) for all UEs
    sigma2_P_MMSE = np.sum(scaling_P_MMSE, axis=0).T

    # Compute the SE in (5.8) for P-MMSE combining
    SE_UatF_PMMSE_DCC[:, n] = prelogFactor * np.log2(1 + bk_P_MMSE * p_full /
                                                     (ck_P_MMSE.conjugate().T @ p_full + sigma2_P_MMSE))

# FIGURES
# Plot Figure 5.4(b)
plt.plot(np.sort(SE_MMSE_original.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k-', linewidth=2)
plt.plot(np.sort(SE_MMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'r-.', linewidth=2)
plt.plot(np.sort(SE_PMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=2)
plt.plot(np.sort(SE_PRZF_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b--', linewidth=2)
plt.plot(np.sort(SE_MR_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=3)
plt.xlabel('Spectral efficiency [bit/s/Hz]', fontsize=12)
plt.ylabel('CDF', fontsize=12)
plt.legend(['MMSE (All)', 'MMSE (DCC)', 'P-MMSE (DCC)', 'P-RZF (DCC)', 'MR (DCC)'], loc='lower right', fontsize=10)
plt.xlim([0, 12])
plt.show()

# Plot Figure 5.5
plt.figure()
plt.plot(np.sort(SE_PMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b', linewidth=2)
plt.plot(np.sort(SE_Genie_PMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'r-.', linewidth=2)
plt.plot(np.sort(SE_UatF_PMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=2)
plt.xlabel('Spectral efficiency [bit/s/Hz]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.legend(['P-MMSE', 'P-MMSE (Genie-aided)', 'P-MMSE (UatF)'], fontsize=16)
plt.xlim([0, 12])
plt.show()

# Plot Figure 5.6(b)
plt.figure()
plt.plot(np.sort(SE_opt_LMMSE_original.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k-', linewidth=2)
plt.plot(np.sort(SE_opt_LMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'r-.', linewidth=2)
plt.plot(np.sort(SE_nopt_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=2)
plt.plot(np.sort(SE_nopt_MR_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b--', linewidth=2)
plt.xlabel('Spectral efficiency [bit/s/Hz]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.legend(['opt LSFD, L-MMSE (All)', 'opt LSFD, L-MMSE (DCC)', 'n-opt LSFD, LP-MMSE (DCC)', 'n-opt LSFD, MR (DCC)'], fontsize=16)
plt.xlim([0, 12])
plt.show()

# Plot Figure 5.7
plt.figure()
plt.plot(np.sort(SE_nopt_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k-', linewidth=2)
plt.plot(np.sort(SE_Genie_nopt_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'r-.', linewidth=2)
plt.plot(np.sort(SE_nopt_MR_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=2)
plt.plot(np.sort(SE_Genie_nopt_MR_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b--', linewidth=2)
plt.xlabel('Spectral efficiency [bit/s/Hz]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.legend(['n-opt LSFD, LP-MMSE', 'n-opt LSFD, LP-MMSE (Genie-aided)', 'n-opt LSFD, MR', 'n-opt LSFD, MR (Genie-aided)'],
           fontsize=16)
plt.xlim([0, 12])
plt.show()

# Plot Figure 5.8
plt.figure()
plt.plot(np.sort(SE_nopt_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b-', linewidth=2)
plt.plot(np.sort(SE_LMMSE_original.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k-', linewidth=2)
plt.plot(np.sort(SE_LPMMSE_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'r-.', linewidth=2)
plt.plot(np.sort(SE_Dist_MR_original.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'k:', linewidth=2)
plt.plot(np.sort(SE_Dist_MR_DCC.flatten()), np.linspace(0, 1, K * nbrOfSetups), 'b--', linewidth=2)
plt.xlabel('Spectral efficiency [bit/s/Hz]', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.legend(['n-opt LSFD, LP-MMSE', 'L-MMSE (All)', 'LP-MMSE (DCC)', 'MR (All)', 'MR (DCC)'],
           fontsize=16)
plt.xlim([0, 12])
plt.show()


print('end')



