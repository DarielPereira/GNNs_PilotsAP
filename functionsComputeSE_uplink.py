import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random
import math

def functionComputeSE_uplink(Hhat, H, D, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p):
    """Compute uplink SE for different receive combining schemes using the capacity bound in Theorem 5.1
    for the centralized scheme and the capacity bound in Theorem  5.4 for the distributed schemes. Compute
    the genie-aided uplink SE from Corollary 5.9 for the centralized operation and 5.10 for ths distributed one.

    INPUT>
    :param Hhat: Matrix with dimension L*N x nbrOfRealizations x k where (:, n, k) is the estimated collective channel
                        to UE k in channel realization n.
    :param H: Matrix with dimension L*N x nbrOfRealizations x k where (:, n, k) is the true collective channel
                        to UE k in channel realization n.
    :param D: DCC matrix with dimensions LxK where the element (l,k) equals '1' if AP l serves
                        UE k, and '0' otherwise
    :param C: matrix with dimension N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel estimation
                        error between AP l and UE k (normalized by noise variance)
    :param tau_c: length of the coherence block
    :param tau_p: length of pilot sequences
    :param nbrOfRealizations: number of channel realizations
    :param N: number of antennas per AP
    :param K: number of UEs
    :param L: number of APs
    :param p: uplink transmit power per UE (same for everyone)
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    :param pilotIndex: matrix with dimensions Kx1 containing the integer indexes of the pilots
                            assigned to the UEs

    OUTPUT>
    --centralized combining--
    SE_MMSE: SEs achieved with MMSE combining
    SE_P_MMSE: SEs achieved with P-MMSE combining
    SE_P_RZF: SEs achieved with p-RZF combining
    SE_MR_cent: SEs achieved with centralized MR combining
    Gen_SE_P_MMSE: Genie-aided SEs achieved with P-MMSE combining
    Gen_SE_P_RZF: Genie-aided SEs achieved with RZF combining
    --distributed combining--
    SE_opt_L_MMSE: SE achieved with optimal LSFD and L-MMSE combining
    SE_nopt_LP_MMSE: SE achieved with near-optimal LSFD and LP-MMSE combining
    SE_nopt_MR: SEs achieved with near-optimal LSFD and local MR combining
    SE_L_MMSE: SEs achieved with L-MMSE combining without LSFD
    SE_MR_dist: SEs achieved with local MR combining without LSFD
    Gen_SE_LP_MMSE: Genie-aided SEs achieved with n-opt LSFD and LP-MMSE combining
    Gen_SE_MR_dist: Genie-aided SEs achieved with n-opt LSFD and distributed MR combining
    --small cell--
    SE_small_MMSE: SEs achieved with L-MMSE combining for small-cell setup
    Gen_SE_small_MMSE: Genie-aided SEs achieved with L-MMSE combining for small-cell setup
    """

    # Store the N x N identity matrix
    eyeN = np.identity(N)



    # Compute the prelog factor assuming only uplink data transmission
    prelogFactor = math.ceil((tau_c-tau_p)/3)/tau_c


    # Prepare to store simulation results
    SE_MMSE = np.zeros((K, 1), dtype=complex)
    SE_P_MMSE = np.zeros((K, 1), dtype=complex)
    SE_P_RZF = np.zeros((K, 1), dtype=complex)
    SE_MR = np.zeros((K, 1), dtype=complex)

    # Go through all channel realizations
    for n in range(nbrOfRealizations):

        # Consider the centralized schemes (Combining vectors and SEs)
        # Go through all UEs
        for k in range(K):

            # Determine the set of serving APs for UE k
            servingAPs, = np.where(D[:, k] == 1)                # cell-free setup

            # Compute the number of APs that serve UE k
            La = len(servingAPs)

            if La > 0:

                # Determine which UEs are served by partially the same set of APs as UE k
                servedUEs = np.sum(D[servingAPs, :], axis=0) >= 1

                # Extract channel realizations and estimation error correlation matrices for the APs involved
                # in the service of UE k
                Hallj_active = np.zeros((N*La, K), dtype=complex)
                Hhatallj_active = np.zeros((N*La, K), dtype=complex)
                C_tot_blk = np.zeros((N*La, N*La), dtype=complex)
                C_tot_blk_partial = np.zeros((N*La, N*La), dtype=complex)

                for l in range(La):
                    Hallj_active[l*N:(l+1)*N, :] = H[servingAPs[l]*N:(servingAPs[l]+1)*N, n, :].reshape(N, K)
                    Hhatallj_active[l * N:(l + 1) * N, :] = Hhat[servingAPs[l] * N:(servingAPs[l] + 1) * N, n, :].reshape(N, K)
                    C_tot_blk[l * N: (l+1) * N, l * N: (l+1) * N] = np.sum(C[:, :, servingAPs[l], :], 2)
                    # # Use this when working with global CSI and knowledge about the AP assignment of some UEs
                    # C_tot_blk_partial[l * N: (l + 1) * N, l * N: (l + 1) * N] = np.sum(C[:, :, servingAPs[l], servedUEs], 2)
                    # Use this when working with local CSI and no knowledge about the AP assignment of the UEs
                    C_tot_blk_partial[l * N: (l + 1) * N, l * N: (l + 1) * N] = np.sum(C[:, :, servingAPs[l], :], 2)

                # Compute MMSE combining according to 5.11
                v = p * (alg.inv(p * (Hhatallj_active @ Hhatallj_active.conjugate().T) +
                                 p * C_tot_blk_partial + np.identity(La * N)) @ Hhatallj_active[:, k])

                # Compute numerator and denominator of instantaneous SINR in 5.5
                numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k]) ** 2
                denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active) ** 2 + v.conjugate().T @ (
                        p * C_tot_blk + np.identity(La * N)) @ v - numerator

                # Update the SE by computing the instantaneous reward for one channel realization
                # according to 5.4
                SE_MMSE[k] = SE_MMSE[k] + prelogFactor * (np.log2(1 + numerator / denominator)).real / nbrOfRealizations


                # Compute P-RZF combining according to 5.18
                # # Use this when working with global CSI and knowledge about the AP assignment of some UEs
                # v = p * (alg.inv(
                #     p * (Hhatallj_active[:, servedUEs] @ Hhatallj_active[:, servedUEs].conjugate().T) + np.identity(
                #         La * N)) @ Hhatallj_active[:, k])

                # Use this when working with local CSI and no knowledge about the AP assignment of the UEs
                v = p * (alg.inv(
                    p * (Hhatallj_active @ Hhatallj_active.conjugate().T) + np.identity(
                        La * N)) @ Hhatallj_active[:, k])

                # Compute numerator and denominator of instantaneous SINR in 5.5
                numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k]) ** 2
                denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active) ** 2 + v.conjugate().T @ (
                        p * C_tot_blk + np.identity(La * N)) @ v - numerator


                # Update the SE by computing the instantaneous reward for one channel realization
                # according to 5.4
                SE_P_RZF[k] = SE_P_RZF[k] + prelogFactor * (
                    np.log2(1 + numerator / denominator)).real / nbrOfRealizations



                # Compute centralized MR combining according to 5.14
                v = Hhatallj_active[:, k]

                # Compute numerator and denominator of instantaneous SINR in 5.5
                numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k]) ** 2
                denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active) ** 2 + v.conjugate().T @ (
                        p * C_tot_blk + np.identity(La * N)) @ v - numerator

                # Update the SE by computing the instantaneous SE for one channel realization according to 5.4
                SE_MR[k] = SE_MR[k] + prelogFactor * (
                    np.log2(1 + numerator / denominator)).real / nbrOfRealizations


                # Compute P-MMSE combining according 5.16
                # # Use this when working with global CSI and knowledge about the AP assignment of some UEs
                # v = p * (alg.inv(p * (Hhatallj_active[:, servedUEs] @ Hhatallj_active[:, servedUEs].conjugate().T) +
                #                  p * C_tot_blk_partial + np.identity(La * N)) @ Hhatallj_active[:, k])

                # Use this when working with local CSI and no knowledge about the AP assignment of the UEs
                v = p * (alg.inv(p * (Hhatallj_active @ Hhatallj_active.conjugate().T) +
                                 p * C_tot_blk_partial + np.identity(La * N)) @ Hhatallj_active[:, k])

                # Compute numerator and denominator of instantaneous SINR in 5.5
                numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k]) ** 2
                denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active) ** 2 + v.conjugate().T @ (
                        p * C_tot_blk + np.identity(La * N)) @ v - numerator


                # Update the SE by computing the instantaneous reward for one channel realization
                # according to 5.4
                SE_P_MMSE[k] = SE_P_MMSE[k] + prelogFactor * (
                    np.log2(1 + numerator / denominator)).real / nbrOfRealizations



            else:
                SE_MMSE[k] = SE_MMSE[k] + 0
                SE_P_RZF[k] = SE_P_RZF[k] + 0
                SE_MR[k] = SE_MR[k] + 0
                SE_P_MMSE[k] = SE_P_MMSE[k] + 0

    return SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE