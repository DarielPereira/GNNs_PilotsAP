import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random

def functionComputeSE_uplink(Hhat, H, D, D_small, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex):
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
    :param D_small: DCC matrix with dimensions LxK where the element (l,k) equals '1' if AP l serves
                        UE k, and '0' otherwise (for small-cell setups)
    :param B: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel estimate
                        between AP l and UE k (normalized by noise variance)
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
    prelogFactor = (1-(tau_p/tau_c))

    # Prepare to store simulation results
    SE_MMSE = np.zeros((K, 1), dtype=complex)
    SE_P_MMSE = np.zeros((K, 1), dtype=complex)
    SE_P_RZF = np.zeros((K, 1), dtype=complex)
    SE_MR_cent = np.zeros((K, 1), dtype=complex)
    SE_opt_L_MMSE = np.zeros((K, 1), dtype=complex)
    SE_nopt_LP_MMSE = np.zeros((K, 1), dtype=complex)
    SE_nopt_MR = np.zeros((K, 1), dtype=complex)
    SE_L_MMSE = np.zeros((K, 1), dtype=complex)
    SE_LP_MMSE = np.zeros((K, 1), dtype=complex)
    SE_MR_dist = np.zeros((K, 1), dtype=complex)
    Gen_SE_P_MMSE = np.zeros((K, 1), dtype=complex)
    Gen_SE_P_RZF = np.zeros((K, 1), dtype=complex)
    Gen_SE_LP_MMSE = np.zeros((K, 1), dtype=complex)
    Gen_SE_MR_dist = np.zeros((K, 1), dtype=complex)
    SE_small_MMSE = np.zeros((K, 1), dtype=complex)
    Gen_SE_small_MMSE = np.zeros((K, 1), dtype=complex)

    # Prepare to store the terms that appear in the SE expressions
    gki_MR = np.zeros((K, L, K), dtype=complex)
    gki2_MR = np.zeros((K, L, K), dtype=complex)
    Fk_MR = np.zeros((L, K), dtype=complex)

    gki_L_MMSE = np.zeros((K, K, L), dtype=complex)
    gki2_L_MMSE = np.zeros((K, K, L), dtype=complex)
    Fk_L_MMSE = np.zeros((L, K), dtype=complex)

    gki_LP_MMSE = np.zeros((K, K, L), dtype=complex)
    gki2_LP_MMSE = np.zeros((K, K, L), dtype=complex)
    Fk_LP_MMSE = np.zeros((L, K), dtype=complex)

    gen_gki_MR = np.zeros((K, L, K, nbrOfRealizations), dtype=complex)
    gen_Fk_MR = np.zeros((L, K, nbrOfRealizations), dtype=complex)

    gen_gki_L_MMSE = np.zeros((K, L, K, nbrOfRealizations), dtype=complex)
    gen_Fk_L_MMSE = np.zeros((L, K, nbrOfRealizations), dtype=complex)

    gen_gki_LP_MMSE = np.zeros((K, L, K, nbrOfRealizations), dtype=complex)
    gen_Fk_LP_MMSE = np.zeros((L, K, nbrOfRealizations), dtype=complex)

    # Compute MR closed-form expectations according to Corollary 5.6
    # Go through each AP
    for l in range(L):
        # extract the users served by AP l
        servedUEs, = np.where(D[l,:]==1)

        # Go through all the UEs served by the AP l
        for k in servedUEs:

            # Noise scaling according to 5.35
            Fk_MR[l, k] = np.trace(B[:, :, l, k])

            for i in range(K):
                # Compute the first term in 5.34
                gki2_MR[i, l, k] = np.trace(B[:, :, l, k]@R[:, :, l, i]).real

                # if UE i shares the same pilot with UE k
                if pilotIndex[k] == pilotIndex[i]:

                    # Term in 5.33
                    gki_MR[i, l, k] = np.trace(B[:, :, l, k]@alg.inv(R[:, :, l, k])@R[:, :, l, i]).real
                    # Second term in 5.34
                    gki2_MR[i, l, k] = gki2_MR[i, l, k] + (np.trace(B[:, :, l, k]@alg.inv(R[:, :, l, k])@R[:, :, l, i]).real)**2

    # Go through all channel realizations
    for n in range(nbrOfRealizations):

        # Consider the centralized schemes (Combining vectors and SEs)
        # Go through all UEs
        for k in range(K):

            # Determine the set of serving APs for UE k
            servingAPs, = np.where(D[:, k] == 1)                # cell-free setup
            servingAP_small, = np.where(D_small[:, k] == 1)     # small-cell setup

            # Compute the number of APs that serve UE k
            La = len(servingAPs)

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
                C_tot_blk_partial[l * N: (l + 1) * N, l * N: (l + 1) * N] = np.sum(C[:, :, servingAPs[l], servedUEs], 2)


            # Compute P-MMSE combining according 5.16
            v = p * (alg.inv(p * (Hhatallj_active[:, servedUEs] @ Hhatallj_active[:, servedUEs].conjugate().T) +
                      p * C_tot_blk_partial + np.identity(La * N)) @ Hhatallj_active[:, k])

            # Compute numerator and denominator of instantaneous SINR in 5.5
            numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k])**2
            denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active)**2 + v.conjugate().T @ (
                    p * C_tot_blk +np.identity(La * N)) @ v -numerator

            # Compute numerator and denominator of instantaneous SINR in 5.50
            numerator_gen = p * np.abs(v.conjugate().T @ Hallj_active[:, k])**2
            denominator_gen = p * alg.norm(v.conjugate().T @ Hallj_active) ** 2 + v.conjugate().T @ v - numerator_gen

            # Update the SE by computing the instantaneous reward for one channel realization
            # according to 5.4
            SE_P_MMSE[k] = SE_P_MMSE[k] + prelogFactor * (np.log2(1+numerator/denominator)).real/nbrOfRealizations

            # Update the SE by computing the instantaneous reward for one channel realization
            # according to 5.49
            Gen_SE_P_MMSE[k] = Gen_SE_P_MMSE[k] + prelogFactor * (np.log2(1 + numerator_gen / denominator_gen)).real / nbrOfRealizations


            # Compute P-RZF combining according to 5.18
            v = p * (alg.inv(p * (Hhatallj_active[:, servedUEs] @ Hhatallj_active[:, servedUEs].conjugate().T) + np.identity(La * N)) @ Hhatallj_active[:, k])

            # Compute numerator and denominator of instantaneous SINR in 5.5
            numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k]) ** 2
            denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active) ** 2 + v.conjugate().T @ (
                    p * C_tot_blk + np.identity(La * N)) @ v - numerator

            # Compute numerator and denominator of instantaneous SINR in 5.50
            numerator_gen = p * np.abs(v.conjugate().T @ Hallj_active[:, k]) ** 2
            denominator_gen = p * alg.norm(v.conjugate().T @ Hallj_active) ** 2 + v.conjugate().T @ v - numerator_gen

            # Update the SE by computing the instantaneous reward for one channel realization
            # according to 5.4
            SE_P_RZF[k] = SE_P_RZF[k] + prelogFactor * (np.log2(1 + numerator / denominator)).real / nbrOfRealizations

            # Update the SE by computing the instantaneous reward for one channel realization
            # according to 5.49
            Gen_SE_P_RZF[k] = Gen_SE_P_RZF[k] + prelogFactor * (
                np.log2(1 + numerator_gen / denominator_gen)).real / nbrOfRealizations


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

            # Extract the required channel estimates and estimation correlation matrices for the SE computation
            # of UE k in small-cell setup
            Hhatallj_active_small = (Hhat[servingAP_small[0] * N:(servingAP_small[0] + 1) * N, n, :]).reshape(N, K)
            C_tot_blk_small = (np.sum(C[:, :, servingAP_small[0], :], 2)).reshape(N, N)

            # Compute L-MMSE combining for small-cell setup according 5.29
            v_small = p * (alg.inv(p * (Hhatallj_active_small @ Hhatallj_active_small.conjugate().T) +
                             p * C_tot_blk_small + np.identity(N)) @ Hhatallj_active_small[:, k])

            # Compute numerator and denominator of instantaneous SINR in 5.5 for L-MMSE combining
            # in small-cell setup
            numerator = p * np.abs(v_small.conjugate().T @ Hhatallj_active_small[:, k]) ** 2
            denominator = p * alg.norm(v_small.conjugate().T @ Hhatallj_active_small) ** 2 + v_small.conjugate().T @ (
                    p * C_tot_blk_small + np.identity(N)) @ v_small - numerator

            # Update the SE by computing the instantaneous reward for one channel realization
            # according to 5.4
            SE_small_MMSE[k] = SE_small_MMSE[k] + prelogFactor * (np.log2(1 + numerator / denominator)).real / nbrOfRealizations


            # Compute centralized MR combining according to 5.14
            v = Hhatallj_active[:, k]

            # Compute numerator and denominator of instantaneous SINR in 5.5
            numerator = p * np.abs(v.conjugate().T @ Hhatallj_active[:, k]) ** 2
            denominator = p * alg.norm(v.conjugate().T @ Hhatallj_active) ** 2 + v.conjugate().T @ (
                    p * C_tot_blk + np.identity(La * N)) @ v - numerator

            # Update the SE by computing the instantaneous SE for one channel realization according to 5.4
            SE_MR_cent[k] = SE_MR_cent[k] + prelogFactor * (np.log2(1 + numerator / denominator)).real / nbrOfRealizations

        # Consider the distributed schemes (Combining vectors)
        # Go through all the APs
        for l in range(L):
            # Extract channel realizations from all UEs to AP l
            Hallj = H[l * N: (l + 1) * N, n, :].reshape(N, K)

            # Extract channel estimates from all UEs to AP l
            Hhatallj = Hhat[l * N: (l + 1) * N, n, :].reshape(N, K)

            # Extract which UEs are served by AP l
            servedUEs, = np.where(D[l, :] == 1)

            # Compute sum of the estimation error covariance matrices of the UEs served by AP l
            Cserved = np.sum(C[:, :, l, servedUEs], axis=2)

            # Compute MR combining according to 5.32
            V_MR = Hhatallj[:, servedUEs]

            # Compute L-MMSE combining according to 5.29
            V_L_MMSE = p * (alg.inv(
                p * (Hhatallj @ Hhatallj.conjugate().T + np.sum(C[:, :, l, :], axis=2)) + eyeN) @ V_MR)

            # Compute LP-MMSE combining according to 5.39
            V_LP_MMSE = p * (alg.inv(p * (V_MR @ V_MR.conjugate().T + Cserved) + eyeN) @ V_MR)

            # Compute the conjugates of the vectors g_{ki} in 5.23 for the combining schemes above for the considerd
            # channel realizations
            TemporMatr_MR = Hallj.conjugate().T @ V_MR
            TemporMatr_L_MMSE = Hallj.conjugate().T @ V_L_MMSE
            TemporMatr_LP_MMSE = Hallj.conjugate().T @ V_LP_MMSE

            # Update the sample mean estimates of the expectations in 5.27
            Fk_L_MMSE[l, servedUEs] = Fk_L_MMSE[l, servedUEs] + alg.norm(V_L_MMSE, axis=0) ** 2 / nbrOfRealizations
            Fk_LP_MMSE[l, servedUEs] = Fk_LP_MMSE[l, servedUEs] + alg.norm(V_LP_MMSE,
                                                                           axis=0) ** 2 / nbrOfRealizations

            # Store the instantaneous combining vector norms of the channel realization n to be used later
            gen_Fk_MR[l, servedUEs, n] = alg.norm(V_MR, axis=0) ** 2
            gen_Fk_L_MMSE[l, servedUEs, n] = alg.norm(V_L_MMSE, axis=0) ** 2
            gen_Fk_LP_MMSE[l, servedUEs, n] = alg.norm(V_LP_MMSE, axis=0) ** 2

            # Update the sample mean estimates of the expectations related to g_{ki} in 5.23 to be used in the SE
            # expression of Theorem 5.4
            gki_L_MMSE[:, servedUEs, l] = gki_L_MMSE[:, servedUEs, l] + TemporMatr_L_MMSE / nbrOfRealizations
            gki_LP_MMSE[:, servedUEs, l] = gki_LP_MMSE[:, servedUEs, l] + TemporMatr_LP_MMSE / nbrOfRealizations

            gki2_L_MMSE[:, servedUEs, l] = gki2_L_MMSE[:, servedUEs, l] + np.abs(
                TemporMatr_L_MMSE) ** 2 / nbrOfRealizations
            gki2_LP_MMSE[:, servedUEs, l] = gki2_LP_MMSE[:, servedUEs, l] + np.abs(
                TemporMatr_LP_MMSE) ** 2 / nbrOfRealizations

            # Store the instantaneous entries of g_{ki} in 5.23 for the channel realization n to be used later
            gen_gki_MR[:, l, servedUEs, n] = TemporMatr_MR
            gen_gki_L_MMSE[:, l, servedUEs, n] = TemporMatr_L_MMSE
            gen_gki_LP_MMSE[:, l, servedUEs, n] = TemporMatr_LP_MMSE

    # Permute the arrays that consist of the expectations that appear in Theorem 5.4 to compute LSFD vectors
    # and the corresponding SEs
    gki_L_MMSE = np.transpose(gki_L_MMSE, (0, 2, 1))
    gki_LP_MMSE = np.transpose(gki_LP_MMSE, (0, 2, 1))
    gki2_L_MMSE = np.transpose(gki2_L_MMSE, (0, 2, 1))
    gki2_LP_MMSE = np.transpose(gki2_LP_MMSE, (0, 2, 1))

    # Prepare to store n-opt LSFD vectors to be used later
    a_nopt1 = np.zeros((L, K), dtype=complex)
    a_nopt2 = np.zeros((L, K), dtype=complex)

    # Compute the SEs for Distributed case
    for k in range(K):

        # Determine the set of serving APs for UEk
        servingAPs, = np.where(D[:, k] == 1)  # cell-free setup

        # Compute the number of APs that serve UE k
        La = len(servingAPs)

        # Determine which UEs are served by partially the same set of APs as UE k
        servedUEs, = np.where(np.sum(D[servingAPs, :], axis=0) >= 1)


        # Expected value of g_{kk} scaled by /sqrt(p) for L-MMSE combining
        num_vector = np.sqrt(p) * gki_L_MMSE[k, servingAPs, k].reshape(-1, 1).conjugate()

        # Compute the matrix whose inverse is taken in 5.30 using the first- and second-order moments
        # of the entries in the vectors g_{ki}
        temporMatr = gki_L_MMSE[:, servingAPs, k].conjugate().T @ gki_L_MMSE[:, servingAPs, k]
        denom_matrix = p * (np.diag(np.sum(gki2_L_MMSE[:, servingAPs, k], axis=0)) + temporMatr -
                            np.diag(np.diag(temporMatr))) - num_vector @ num_vector.conjugate().T \
                       + np.diag(Fk_L_MMSE[servingAPs, k])

        # Compute the opt LSFD according 5.30
        a_opt = alg.inv(denom_matrix) @ num_vector

        # Compute the weights for the case without LSFD
        a_dist = np.ones((La, 1))

        # Compute the SE achieved with opt LSFD and L-MMSE combining according to 5.25
        SE_opt_L_MMSE[k] =  prelogFactor * (np.log2(1 + np.abs(a_opt.conjugate().T @ num_vector)**2/
                                                    (a_opt.conjugate().T @ denom_matrix @ a_opt))).real

        # Compute the SE achieved with without LSFD and with L-MMSE combining according to 5.25
        SE_L_MMSE[k] = prelogFactor * (np.log2(1 + np.abs(a_dist.conjugate().T @ num_vector) ** 2 /
                                                   (a_dist.conjugate().T @ denom_matrix @ a_dist))).real


        # Expected value of g_{kk} scaled by /sqrt(p) for LP-MMSE combining
        num_vector = np.sqrt(p) * gki_LP_MMSE[k, servingAPs, k].reshape(-1, 1).conjugate()

        # Compute the denominator matrix to compute SE in theorem 5.4 using the first- and second-order moments
        # of the entries in the vectors g_{ki}
        temporMatr = gki_LP_MMSE[:, servingAPs, k].conjugate().T @ gki_LP_MMSE[:, servingAPs, k]
        denom_matrix = p * (np.diag(np.sum(gki2_LP_MMSE[:, servingAPs, k], axis=0)) + temporMatr -
                            np.diag(np.diag(temporMatr))) - num_vector @ num_vector.conjugate().T \
                       + np.diag(Fk_LP_MMSE[servingAPs, k])

        # Compute the matrix whose inverse is taken in 5.41 using the first- and second-order moments of the
        # entries in the vectors g_{ki}
        temporMatr = gki_LP_MMSE[:, servingAPs, k][servedUEs, :].conjugate().T @ gki_LP_MMSE[:, servingAPs, k][servedUEs, :]
        denom_matrix_partial = p * (np.diag(np.sum(gki2_LP_MMSE[:, servingAPs, k][servedUEs, :], axis=0)) + temporMatr -
                            np.diag(np.diag(temporMatr))) - num_vector @ num_vector.conjugate().T \
                       + np.diag(Fk_LP_MMSE[servingAPs, k])

        # Compute the n-opt LSFD according to 5.41 for LP-MMSE
        a_nopt = alg.inv(denom_matrix_partial) @ num_vector

        # Compute the SE achieved with n-opt LSFD and LP-MMSE combining according to 5.25
        SE_nopt_LP_MMSE[k] = prelogFactor * (np.log2(1 + np.abs(a_nopt.conjugate().T @ num_vector) ** 2 /
                                                    (a_nopt.conjugate().T @ denom_matrix @ a_nopt))).real

        # Compute the SE achieved without LSFD and with LP-MMSE combining according to 5.25
        SE_LP_MMSE[k] = prelogFactor * (np.log2(1 + np.abs(a_dist.conjugate().T @ num_vector) ** 2 /
                                               (a_dist.conjugate().T @ denom_matrix @ a_dist))).real

        # Store the n-opt LSFD vector for LP-MMSE combining to be used later
        a_nopt1[servingAPs, k] = a_nopt.flatten()


        # Expected value of g_{kk}, scaled by /sqrt{p} for local MR combining
        num_vector = np.sqrt(p) * gki_MR[k, servingAPs, k].reshape(-1, 1).conjugate()

        # Compute the denominator matrix to compute SE in theorem 5.4 using the first- and second-order moments
        # of the entries in the vectors g_{ki}
        temporMatrrr = gki_MR[:, servingAPs, k].conjugate().T @ gki_MR[:, servingAPs, k]
        denom_matrix = p * (np.diag(np.sum(gki2_MR[:, servingAPs, k], axis=0)) + temporMatrrr -
                            np.diag(np.diag(temporMatrrr))) - num_vector @ num_vector.conjugate().T \
                       + np.diag(Fk_MR[servingAPs, k])

        # Compute the matrix whose inverse is taken in 5.41 using the first- and second-order moments of the
        # entries in the vectors g_{ki}
        temporMatrrr = gki_MR[:, servingAPs, k][servedUEs, :].conjugate().T @ gki_MR[:, servingAPs, k][
                                                                                 servedUEs, :]
        denom_matrix_partial = p * (np.diag(np.sum(gki2_MR[:, servingAPs, k][servedUEs, :], axis=0)) + temporMatrrr -
                                    np.diag(np.diag(temporMatrrr))) - num_vector @ num_vector.conjugate().T \
                               + np.diag(Fk_MR[servingAPs, k])

        # Compute the n-opt LSFD according to 5.41 for local MR combining
        a_nopt = alg.inv(denom_matrix_partial) @ num_vector

        # Compute the SE achieved with n-opt LSFD and Local MR combining according to 5.25
        SE_nopt_MR[k] = prelogFactor * (np.log2(1 + np.abs(a_nopt.conjugate().T @ num_vector) ** 2 /
                                                     (a_nopt.conjugate().T @ denom_matrix @ a_nopt))).real

        # Compute the SE achieved without LSFD and with local MR combining
        SE_MR_dist[k] = prelogFactor * (np.log2(1 + np.abs(a_dist.conjugate().T @ num_vector) ** 2 /
                                                (a_dist.conjugate().T @ denom_matrix @ a_dist))).real

        # Store the n-opt LSFD vector for local MR combining to be used later
        a_nopt2[servingAPs, k] = a_nopt.flatten()


    # Go through all channel realizations
    for n in range(nbrOfRealizations):

        # Go through all UEs
        for k in range(K):

            # Determine the set of serving APs
            servingAPs, = np.where(D[:, k] == 1)                # cell-free setup
            servingAP_small, = np.where(D_small[:, k] == 1)     # small-cell setup

            # Compute the numerator and the denominator in 5.53 with a single serving AP in a small-cell setup
            numerator_gen = p * np.abs(gen_gki_L_MMSE[k,servingAP_small, k, n])**2
            denominator_gen = p * gen_gki_L_MMSE[:, servingAP_small, k, n].conjugate().T @ \
                              gen_gki_L_MMSE[:, servingAP_small, k, n] + gen_Fk_L_MMSE[servingAP_small, k, n] \
                              - numerator_gen

            # Update the genie-aided SE by computing the instantaneous SE for one channel realization
            # according to 5.52 in small-cell setup
            Gen_SE_small_MMSE[k] = Gen_SE_small_MMSE[k] + prelogFactor \
                                   * (np.log2(1 + numerator_gen/denominator_gen)).real/nbrOfRealizations

            # Compute the numerator and the denominator in 5.53 for n-opt LSFD and LP-MMSE combining
            numerator_gen = p * np.abs(a_nopt1[servingAPs, k].conjugate().T @
                                       (gen_gki_LP_MMSE[k, servingAPs, k, n]).conjugate().reshape(-1, 1)) ** 2
            temporMatrrr = gen_gki_LP_MMSE[:, servingAPs, k, n].conjugate().T @ gen_gki_LP_MMSE[:, servingAPs, k, n]
            denominator_gen = p * a_nopt1[servingAPs, k].conjugate().T @ temporMatrrr @ a_nopt1[servingAPs, k] \
                              + a_nopt1[servingAPs, k].conjugate().T @ np.diag(gen_Fk_LP_MMSE[servingAPs, k, n]) \
                              @ a_nopt1[servingAPs, k] - numerator_gen

            # Update the genie-aided SE by computing the instantaneous SE for one channel realization
            # according to 5.52
            Gen_SE_LP_MMSE[k] = Gen_SE_LP_MMSE[k] + prelogFactor \
                                * (np.log2(1 + numerator_gen/denominator_gen)).real/nbrOfRealizations

            # Compute the numerator and the denominator in 5.53 for n-opt LSFD and local MR combining
            numerator_gen = p * np.abs(a_nopt2[servingAPs, k].conjugate().T @
                                       (gen_gki_MR[k, servingAPs, k, n]).conjugate().reshape(-1, 1)) ** 2
            temporMatrrr = gen_gki_MR[:, servingAPs, k, n].conjugate().T @ gen_gki_MR[:, servingAPs, k, n]
            denominator_gen = p * a_nopt2[servingAPs, k].conjugate().T @ temporMatrrr @ a_nopt2[servingAPs, k] \
                              + a_nopt2[servingAPs, k].conjugate().T @ np.diag(gen_Fk_MR[servingAPs, k, n]) \
                              @ a_nopt2[servingAPs, k] - numerator_gen

            # Update the genie-aided SE by computing the instantaneous SE for one channel realization according to 5.52
            Gen_SE_MR_dist[k] = Gen_SE_MR_dist[k] + prelogFactor \
                                * (np.log2(1 + numerator_gen / denominator_gen)).real / nbrOfRealizations


    SEs_dict = {'SE_MMSE':SE_MMSE, 'SE_P_MMSE':SE_P_MMSE, 'SE_P_RZF':SE_P_RZF, 'SE_MR_cent':SE_MR_cent,
            'SE_opt_L_MMSE':SE_opt_L_MMSE, 'SE_nopt_LP_MMSE':SE_nopt_LP_MMSE, 'SE_nopt_MR':SE_nopt_MR,
            'SE_L_MMSE':SE_L_MMSE, 'SE_LP_MMSE':SE_LP_MMSE, 'SE_MR_dist':SE_MR_dist, 'Gen_SE_P_MMSE':Gen_SE_P_MMSE,
            'Gen_SE_P_RZF':Gen_SE_P_RZF, 'Gen_SE_LP_MMSE':Gen_SE_LP_MMSE, 'Gen_SE_MR_dist':Gen_SE_MR_dist,
            'SE_small_MMSE':SE_small_MMSE, 'Gen_SE_small_MMSE':Gen_SE_small_MMSE}

    return SE_MMSE, SE_P_MMSE, SE_P_RZF, SE_MR_cent, SE_opt_L_MMSE,SE_nopt_LP_MMSE, SE_nopt_MR, SE_L_MMSE, SE_LP_MMSE, \
           SE_MR_dist, Gen_SE_P_MMSE, Gen_SE_P_RZF, Gen_SE_LP_MMSE, Gen_SE_MR_dist, SE_small_MMSE, Gen_SE_small_MMSE