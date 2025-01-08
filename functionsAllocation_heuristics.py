import numpy as np
import itertools
from functionsUtils import db2pow
from functionsComputeSE_uplink import functionComputeSE_uplink



def PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode):
    """Compute the pilot assignment for a set of UEs
    INPUT>
    :param R: matrix with dimensions (N, N, L, K) containing the channel correlation matrices
    :param gainOverNoisedB: matrix with dimensions (L, K) containing the channel gains
    :param tau_p: number of pilots
    :param L: number of APs
    :param K: number of UEs
    :param N: number of antennas at the APs
    :param mode: pilot assignment mode
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # to store pilot assignment
    pilotIndex = -1 * np.ones((K), int)

    # check for PA mode
    match mode:
        case 'random':
            print('implement random')

        case 'DCC':

            # Determine the pilot assignment
            for k in range(0, K):

                # Determine the master AP for UE k by looking for the AP with best channel condition
                master = np.argmax(gainOverNoisedB[:, k])

                if k <= tau_p - 1:  # Assign orthogonal pilots to the first tau_p UEs
                    pilotIndex[k] = k

                else:  # Assign pilot for remaining users

                    # Compute received power to the master AP from each pilot
                    pilotInterference = np.zeros(tau_p)

                    for t in range(tau_p):
                        pilotInterference[t] = np.sum(db2pow(gainOverNoisedB[master, :k][pilotIndex[:k] == t]))

                    # Find the pilot with least received power
                    bestPilot = np.argmin(pilotInterference)
                    pilotIndex[k] = bestPilot

    return pilotIndex


def APassignment(nbrOfRealizations, R, gainOverNoisedB, Hhat, H, B, C, p, tau_c, tau_p, L, N, K, I, pilotIndex,
                 mode, comb_mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # to store AP assignment
    D = np.zeros((L, K))

    # check for PA mode
    match mode:
        case 'random':
            print('implement random')

        case 'BPA':

            # Each AP serves the UE with the strongest channel condition on each of the pilots
            for l in range(L):
                for t in range(tau_p):
                    pilotUEs, = np.where(pilotIndex == t)
                    if len(pilotUEs) > 0:
                        UEindex = np.argmax(gainOverNoisedB[l, pilotIndex == t])
                        D[l, pilotUEs[UEindex]] = 1

        case 'ALL':

            D = np.ones((L, K))

        case 'Graph':

            # Compute the feasible AP assignments
            feasible_APassignments = np.array(list(itertools.product([0, 1], repeat=I)))

            # To Store the best M APs for each UE
            candidateAPs = np.zeros((K, L), int)

            for k in range(K):
                # Get the M best serving APs to the new UE
                candidateAPsindex = np.argsort(gainOverNoisedB[:, k])[-I:][::-1]
                candidateAPs[k, candidateAPsindex] = 1

            for k in range(K):

                candidateAPsindex = np.argsort(gainOverNoisedB[:, k])[-I:][::-1]

                # Get the relevant UEs by assuming a UE is relevant if share at least one best AP
                relevantUEindex = []
                for i in range(K):
                    if np.dot(candidateAPs[k, :], candidateAPs[i, :]) > 0:
                        relevantUEindex.append(i)

                # The number of UEs in the graph on interest
                K_small = len(relevantUEindex)

                # to store pilot assignment
                pilotIndex_small = -1 * np.ones((K_small), int)
                pilotIndex_small[:] = pilotIndex[relevantUEindex]

                # to store AP assignment
                D_small = np.zeros((I, K_small))
                R_small = np.zeros((N, N, I, K_small), dtype=complex)
                Hhat_small = np.zeros((I * N, nbrOfRealizations, K_small), dtype=complex)
                H_small = np.zeros((I * N, nbrOfRealizations, K_small), dtype=complex)
                B_small = np.zeros((R_small.shape), dtype=complex)
                C_small = np.zeros((R_small.shape), dtype=complex)

                # Store all the individual SE values
                SEs = np.zeros((len(feasible_APassignments), K_small))

                # Store all the sum-SE values
                sum_SEs = np.zeros((len(feasible_APassignments)))

                # Go over the best serving APs
                for idx in range(I):
                    # Go over the served UEs
                    for jdx in range(K_small):
                        # Get the reduced versions of R and D matrices
                        R_small[:, :, idx, jdx] = R[:, :, candidateAPsindex[idx], relevantUEindex[jdx]]
                        # D_small[idx, jdx] = D[candidateAPsindex[idx], relevantUEindex[jdx]]
                        Hhat_small[idx * N:(idx + 1) * N, :, jdx] \
                            = Hhat[candidateAPsindex[idx] * N:(candidateAPsindex[idx] + 1) * N, :, relevantUEindex[jdx]]
                        H_small[idx * N:(idx + 1) * N, :, jdx] \
                            = H[candidateAPsindex[idx] * N:(candidateAPsindex[idx] + 1) * N, :, relevantUEindex[jdx]]
                        B_small[:, :, idx, jdx] = B[:, :, candidateAPsindex[idx], relevantUEindex[jdx]]
                        C_small[:, :, idx, jdx] = C[:, :, candidateAPsindex[idx], relevantUEindex[jdx]]

                # Try each AP assignment:
                for idx, APassignment in enumerate(feasible_APassignments):

                    D_small[:, np.where(np.array(relevantUEindex) == k)[0][0]] = APassignment

                    # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                    SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat_small, H_small, D_small,
                                                                                   C_small, tau_c, tau_p,
                                                                                   nbrOfRealizations, N,
                                                                                   K_small, I, p)
                    match comb_mode:
                        case 'MMSE':
                            SE = SE_MMSE
                        case 'P_RZF':
                            SE = SE_P_RZF
                        case 'MR':
                            SE = SE_MR
                        case 'P_MMSE':
                            SE = SE_P_MMSE
                        case _:
                            print('ERROR: Combining mismatching')
                            SE = 0

                    sum_SE = np.sum(SE)

                    sum_SEs[idx] = sum_SE

                    SEs[idx, :] = SE[:].flatten()

                bestAPassignment_index = np.argmax(sum_SEs)

                best_APassignment = feasible_APassignments[bestAPassignment_index]

                assignAPs, = np.where(best_APassignment == 1)
                assingmColumn = np.zeros((L, 1))
                assingmColumn[candidateAPsindex[assignAPs]] = 1
                best_APassignment = assingmColumn
                D[:, k] = best_APassignment.flatten()

        case 'I_best':

            for k in range(K):
                # Get the M best serving APs to the new UE
                candidateAPsindex = np.argsort(gainOverNoisedB[:, k])[-I:][::-1]

                D[candidateAPsindex, k] = 1

    return D