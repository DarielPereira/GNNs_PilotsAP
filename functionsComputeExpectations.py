import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random

def ComputeExpectations(Hhat, H, D, C, nbrOfRealizations, N, K, L, p):
    """Compute expectations that appear in the uplink and downlink SE expressions

    INPUT>
    :param Hhat: Matrix with dimension L*N x nbrOfRealizations x k where (:, n, k) is the estimated collective channel
                        to UE k in channel realization n.
    :param H: Matrix with dimension L*N x nbrOfRealizations x k where (:, n, k) is the true collective channel
                        to UE k in channel realization n.
    :param D: DCC matrix with dimensions LxK where the element (l,k) equals '1' if AP l serves
                        UE k, and '0' otherwise
    :param C: matrix with dimension N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel estimation
                        error between AP l and UE k (normalized by noise variance)
    :param nbrOfRealizations: number of channel realizations
    :param N: number of antennas per AP
    :param K: number of UEs
    :param L: number of APs
    :param p: vector of UE transmit powers

    OUTPUT>
    signal_P_MMSE: Matrix with dimension K x K where (i,k) is the Monte-Carlo estimation of expected value
                        of h_i^HD_kw_k where w_k is P-MMSE combiner/precoder
    signal2_P_MMSE: Matrix with dimension K x K where (i,k) is the Monte-Carlo estimation of expected value
                        of |h_i^HD_kw_k|^2 where w_k is P-MMSE combiner/precoder
    scaling_P_MMSE: Matrix with dimension L x K where (l,k) is the Monte-Carlo estimation of expected value of the norm square of the portion of
                        w_k corresponding to AP l for P-MMSE combiner/precoder if AP l serves UE k, zero otherwise
    signal_P_RZF: Matrix with dimension K x K, organized in the same way as signal_P_MMSE but for
                        P-RZF combining/precoding
    signal2_P_RZF: Matrix with dimension K x K, organized in the same way as signal2_P_MMSE but for
                        P-RZF combining/precoding
    scaling_P_RZF: Matrix with dimension L x K, organized in the same way as scaling_P_MMSE but for
                        P-RZF combining/precoding
    signal_LP_MMSE: Matrix with dimension K x K x L where (i,k,l) is the Monte-Carlo estimation of
                   expected value of h_{il}^HD_{kl}w_{kl} where w_{kl} is LP-MMSE combiner/precoder
    signal2_LP_MMSE: Matrix with dimension K x K x L where (i,k,l) is the Monte-Carlo estimation of
                   expected value of |h_{il}^HD_{kl}w_{kl}|^2 where w_{kl} is LP-MMSE combiner/precode
    scaling_LP_MMSE: Matrix with dimension L x K where (l,k) is the Monte-Carlo estimation of
                   expected value of the norm square of D_{kl}w_{kl} for LP-MMSE combiner/precoder
    """

    # Store the NxN identity matrix
    eyeN = np.identity(N)

    # Obtain the diagonal matrix with UE transmit powers as its diagonal entries
    PowMat = np.diag(p)

    # Scale C by power coefficients
    Cp = np.zeros(C.shape, dtype=complex)
    for k in range(K):
        Cp[:, :, :, k] = p[k] * C[:, :, :, k]

    # Prepare to store the simulation results
    signal_P_MMSE = np.zeros((K, K), dtype=complex)
    signal2_P_MMSE = np.zeros((K, K), dtype=complex)
    scaling_P_MMSE = np.zeros((L, K), dtype=complex)

    signal_P_RZF = np.zeros((K, K), dtype=complex)
    signal2_P_RZF =np.zeros((K, K), dtype=complex)
    scaling_P_RZF = np.zeros((L, K), dtype=complex)

    signal_LP_MMSE = np.zeros((K, K, L), dtype=complex)
    signal2_LP_MMSE = np.zeros((K, K, L), dtype=complex)
    scaling_LP_MMSE = np.zeros((L, K), dtype=complex)

    # Compute scaling factors for combining/precoding

    # Go through all channel realizations
    for n in range(nbrOfRealizations):

        # Go through all APs
        for l in range(L):
            # Extract channel realizations form all UEs to AP l
            Hallj = H[l * N: (l + 1) * N, n, :].reshape(N, K)

            # Extract channel estimates from all UEs to AP l
            Hhatallj = Hhat[l * N: (l + 1) * N, n, :].reshape(N, K)

            # Extract which UEs are served by AP l
            servedUEs, = np.where(D[l, :] == 1)

            # Obtain the statistical matrices used for computing partial combining/precoding schemes
            Cpserved = np.sum(Cp[:, :, l, servedUEs], axis=2).reshape(N, N)
            Pserved = PowMat[servedUEs.reshape(-1,1), servedUEs]

            # Compute MR combining scaled by squared root of transmit powers
            Vp_MR = Hhatallj[:, servedUEs] @ np.sqrt(Pserved)

            # Compute LP-MMSE combining
            V_LP_MMSE = (alg.inv((Vp_MR @ Vp_MR.conjugate().T) + Cpserved + eyeN) @ Vp_MR) @ np.sqrt(Pserved)

            # Go through all UEs served by the AP
            for ind in range(len(servedUEs)):

                # Extract UE index
                k = servedUEs[ind]

                # Normalize LP-MMSE precoding
                w = V_LP_MMSE[:, ind]

                # Compute realizations of the terms inside the expectations of the signal and interference terms
                # in the SE expressions and update Monte-Carlo estimates
                signal2_LP_MMSE[:, k, l] = signal2_LP_MMSE[:, k, l] + (np.abs(Hallj.conjugate().T @ w)**2)/nbrOfRealizations

                signal_LP_MMSE[:, k, l] = signal_LP_MMSE[:, k, l]  + (Hallj.conjugate().T @ w)/nbrOfRealizations

                scaling_LP_MMSE[l, k] = scaling_LP_MMSE[l, k] + np.sum(np.abs(w)**2)/nbrOfRealizations

        # Consider the centralized schemes

        # Go through all UEs
        for k in range(K):

            # Determine the set of serving APs
            servingAPs, = np.where(D[:, k] == 1)  # cell-free setup

            # Compute the number of APs that serve UE k
            La = len(servingAPs)

            # Determine which UEs are served by partially the same set of APs as UE k
            servedUEs = np.sum(D[servingAPs, :], axis=0) >= 1

            # Extract channel realizations and estimation error correlation matrices for the APs
            # involved in the service of UE k
            Hallj_active = np.zeros((N * La, K), dtype=complex)
            Hhatallj_active = np.zeros((N * La, K), dtype=complex)
            Cp_tot_blk = np.zeros((N * La, N * La), dtype=complex)
            Cp_tot_blk_partial = np.zeros((N * La, N * La), dtype=complex)

            for l in range(La):
                Hallj_active[l * N:(l + 1) * N, :] = H[servingAPs[l] * N:(servingAPs[l] + 1) * N, n, :].reshape(N, K)
                Hhatallj_active[l * N:(l + 1) * N, :] = Hhat[servingAPs[l] * N:(servingAPs[l] + 1) * N, n, :].reshape(N,
                                                                                                                      K)
                Cp_tot_blk[l * N: (l + 1) * N, l * N: (l + 1) * N] = np.sum(Cp[:, :, servingAPs[l], :], 2)
                Cp_tot_blk_partial[l * N: (l + 1) * N, l * N: (l + 1) * N] = np.sum(Cp[:, :, servingAPs[l], servedUEs], 2)

            Hphatallj_active = Hhatallj_active @ np.sqrt(PowMat)

            # Compute P-MMSE combining/precoding
            w = (alg.inv((Hphatallj_active[:, servedUEs] @ Hphatallj_active[:, servedUEs].conjugate().T)
                         + Cp_tot_blk_partial+np.identity(La*N)) @ Hphatallj_active[:, k])*np.sqrt(p[k])

            # Compute realizations of the terms inside the expectations of the signal and interference terms in the SE
            # expressions and update Monte-Carlo estimates
            tempor = Hallj_active.conjugate().T @ w

            signal2_P_MMSE[:, k] = signal2_P_MMSE[:, k] + np.abs(tempor)**2/nbrOfRealizations

            signal_P_MMSE[:, k] = signal_P_MMSE[:, k] + tempor/nbrOfRealizations

            for l in range(La):
                # Extract the portions of the P-MMSE combining/precoding vector corresponding to each
                # serving AP and compute the instantaneous norm square of it to update Monte-Carlo estimation
                w2 = w[l * N: (l+1) * N]

                scaling_P_MMSE[servingAPs[l], k] = scaling_P_MMSE[servingAPs[l], k] + np.sum(np.abs(w2)**2) / nbrOfRealizations

            # Compute P-RZF combining/precoding
            w = (alg.inv((Hphatallj_active[:, servedUEs] @ Hphatallj_active[:, servedUEs].conjugate().T) +
                         np.identity(La*N)) @ Hphatallj_active[:, k])*np.sqrt(p[k])

            # Compute realizations of the terms inside the expectations of the signal and interference terms in the
            # SE expressions and update Monte-Carlo estimates
            tempor = Hallj_active.conjugate().T @ w

            signal2_P_RZF[:, k] = signal2_P_RZF[:, k] + np.abs(tempor) ** 2 / nbrOfRealizations

            signal_P_RZF[:, k] = signal_P_RZF[:, k] + tempor / nbrOfRealizations

            for l in range(La):
                # Extract the portions of the P-RZF combining/precoding vector corresponding to each
                # serving AP and compute the instantaneous norm square of it to update Monte-Carlo estimation
                w2 = w[l * N: (l+1) * N]

                scaling_P_RZF[servingAPs[l], k] = scaling_P_RZF[servingAPs[l], k] + np.sum(np.abs(w2)**2) \
                                                  / nbrOfRealizations

    return signal_P_MMSE, signal2_P_MMSE.real, scaling_P_MMSE.real, signal_P_RZF, signal2_P_RZF.real, \
           scaling_P_RZF.real, signal_LP_MMSE,signal2_LP_MMSE.real, scaling_LP_MMSE.real


