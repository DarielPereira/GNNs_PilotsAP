import numpy as np
import itertools
import numpy.linalg as linalg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from functionsUtils import db2pow, localScatteringR, correlationNormalized_grid
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink




def AP_PilotAssignment_UEsBLock(R, gainOverNoisedB, tau_p, L, N, mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    K = gainOverNoisedB.shape[1]

    # to store pilot assignment
    pilotIndex = -1 * np.ones((K), int)

    # to store AP assignment
    D = np.zeros((L, K))

    # check for PA mode
    match mode:
        case 'random':
            print('implement random')

        case 'DCC':

            # Determine the pilot assignment
            for k in range(0, K):

                # Determine the master AP for UE k by looking for the AP with best channel condition
                master = np.argmax(gainOverNoisedB[:, k])

                # Set the master AP as serving AP
                D[master, k] = 1

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

            # Each AP serves the UE with the strongest channel condition on each of the pilots
            for l in range(L):
                for t in range(tau_p):
                    pilotUEs, = np.where(pilotIndex == t)
                    if len(pilotUEs) > 0:
                        UEindex = np.argmax(gainOverNoisedB[l, pilotIndex == t])
                        D[l, pilotUEs[UEindex]] = 1

        case 'ALL':

            # Determine the pilot assignment
            for k in range(0, K):

                # Determine the master AP for UE k by looking for the AP with best channel condition
                master = np.argmax(gainOverNoisedB[:, k])

                # Set the master AP as serving AP
                D[master, k] = 1

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

            D = np.ones((L, K))

    return pilotIndex, D


def AP_Pilot_newUE(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, L, N, M,
                   old_D, old_pilotIndex, comb_mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # Get the M best serving APs to the new UE
    bestAPsindex = np.argsort(gainOverNoisedB[:, -1])[-M:][::-1]

    # compute all the feasible pilot assignments
    feasible_APassignments = np.array(list(itertools.product([0, 1], repeat=M)))

    # Get the UEs being served by each best serving AP
    servedUEs = []
    for l in bestAPsindex:
        servedBYl, = np.where(old_D[l, :] == 1)
        servedUEs.append(servedBYl)

    # Get a list with all the UEs served by these APs
    servedUEindex = list(set({}).union(*servedUEs))

    # The number of UEs in the graph on interest
    K_small = len(servedUEindex) + 1

    # to store pilot assignment
    pilotIndex_small = -1 * np.ones((K_small), int)

    # to store AP assignment
    D_small = np.zeros((M, K_small))
    R_small = np.zeros((N, N, M, K_small), dtype=complex)

    if K_small > 1:
        print('K>1')
        # Go over the best serving APs
        for idx in range(M):
            # Go over the served UEs
            for jdx in range(K_small - 1):
                # Get the reduced versions of R and D matrices
                R_small[:, :, idx, jdx] = R[:, :, bestAPsindex[idx], servedUEindex[jdx]]
                D_small[idx, jdx] = old_D[bestAPsindex[idx], servedUEindex[jdx]]
            # Include the R matrix of the new UE
            R_small[:, :, idx, -1] = R[:, :, bestAPsindex[idx], -1]

        # Get the reduced pilot allocation
        pilotIndex_small[:-1] = old_pilotIndex[servedUEindex]

        # To store the best values
        best_pilot = -1
        best_SE = 0
        best_APassignment = np.zeros(M)

        # Store all the SE
        SEs = np.zeros((tau_p, len(feasible_APassignments)))

        # Run over the pilots
        for t in range(tau_p):

            pilotIndex_small[-1] = t

            # Generate channel realizations with estimates and estimation error matrices
            Hhat, H, B, C = channelEstimates(R_small, nbrOfRealizations, M, K_small, N, tau_p, pilotIndex_small, p)

            # Try each AP assignment:
            for idx, APassignment in enumerate(feasible_APassignments):

                D_small[:, -1] = APassignment

                # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D_small, C, tau_c, tau_p,
                                                            nbrOfRealizations, N, K_small, M, p)
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

                SEs[t, idx] = sum_SE

                APcount = np.sum(APassignment == 1)

        best_SE = np.max(SEs)

        flatten_index = np.argmax(SEs)
        idx_tuple = np.unravel_index(flatten_index, SEs.shape)
        best_APassignment = feasible_APassignments[idx_tuple[1]]
        best_pilot = idx_tuple[0]

    else:

        for idx in range(M):
            R_small[:, :, idx, 0] = R[:, :, bestAPsindex[idx], 0]

        # Store all the SE
        SEs = np.zeros((len(feasible_APassignments)))

        # Set pilot index to the first pilot
        pilotIndex_small[0] = 0
        best_pilot = 0

        # To store values
        best_SE = 0
        best_APassignment = np.zeros(M)

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R_small, nbrOfRealizations, M, K_small, N, tau_p, pilotIndex_small, p)

        # Try each AP assignment:
        for idx, APassignment in enumerate(feasible_APassignments):

            D_small[:, 0] = APassignment

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D_small, C, tau_c, tau_p,
                                                        nbrOfRealizations, N, K_small, M, p)

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

            SEs[idx] = sum_SE
            APcount = np.sum(APassignment == 1)

        best_SE = np.max(SEs)
        best_APassignment = feasible_APassignments[np.argmax(SEs)]


    pilotIndex = np.hstack((old_pilotIndex, [best_pilot]))

    assignAPs, = np.where(best_APassignment == 1)
    assingmColumn = np.zeros((L, 1))
    assingmColumn[bestAPsindex[assignAPs]] = 1
    D = np.hstack((old_D, assingmColumn))

    print('Best AP assigment: ', best_APassignment)
    print('Sum SE: ', best_SE)

    return pilotIndex, D

def AP_Pilot_GeneratingSamples(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, L, N, M,
                   old_D, old_pilotIndex, comb_mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # Get the M best serving APs to the new UE
    bestAPsindex = np.argsort(gainOverNoisedB[:, -1])[-M:][::-1]

    # compute all the feasible pilot assignments
    feasible_APassignments = np.array(list(itertools.product([0, 1], repeat=M)))

    # Get the UEs being served by each best serving AP
    servedUEs = []
    for l in bestAPsindex:
        servedBYl, = np.where(old_D[l, :] == 1)
        servedUEs.append(servedBYl)

    # Get a list with all the UEs served by these APs
    servedUEindex = list(set({}).union(*servedUEs))

    # The number of UEs in the graph on interest
    K_small = len(servedUEindex) + 1

    # to store pilot assignment
    pilotIndex_small = -1 * np.ones((K_small), int)

    # to store AP assignment
    D_small = np.zeros((M, K_small))
    R_small = np.zeros((N, N, M, K_small), dtype=complex)

    if K_small > 1:
        print('K>1')
        # Go over the best serving APs
        for idx in range(M):
            # Go over the served UEs
            for jdx in range(K_small - 1):
                # Get the reduced versions of R and D matrices
                R_small[:, :, idx, jdx] = R[:, :, bestAPsindex[idx], servedUEindex[jdx]]
                D_small[idx, jdx] = old_D[bestAPsindex[idx], servedUEindex[jdx]]
            # Include the R matrix of the new UE
            R_small[:, :, idx, -1] = R[:, :, bestAPsindex[idx], -1]

        # Get the reduced pilot allocation
        pilotIndex_small[:-1] = old_pilotIndex[servedUEindex]

        # To store the best values
        best_pilot = -1
        best_SE = 0
        best_APassignment = np.zeros(M)

        # Store all the SE
        SEs = np.zeros((tau_p, len(feasible_APassignments)))

        # Run over the pilots
        for t in range(tau_p):

            pilotIndex_small[-1] = t

            # Generate channel realizations with estimates and estimation error matrices
            Hhat, H, B, C = channelEstimates(R_small, nbrOfRealizations, M, K_small, N, tau_p, pilotIndex_small, p)

            # Try each AP assignment:
            for idx, APassignment in enumerate(feasible_APassignments):

                D_small[:, -1] = APassignment

                # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
                SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D_small, C, tau_c, tau_p,
                                                            nbrOfRealizations, N, K_small, M, p)
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

                SEs[t, idx] = sum_SE

                APcount = np.sum(APassignment == 1)

        best_SE = np.max(SEs)

        flatten_index = np.argmax(SEs)
        idx_tuple = np.unravel_index(flatten_index, SEs.shape)
        best_APassignment = feasible_APassignments[idx_tuple[1]]
        best_pilot = idx_tuple[0]

    else:

        for idx in range(M):
            R_small[:, :, idx, 0] = R[:, :, bestAPsindex[idx], 0]

        # Store all the SE
        SEs = np.zeros((len(feasible_APassignments)))

        # Set pilot index to the first pilot
        pilotIndex_small[0] = 0
        best_pilot = 0

        # To store values
        best_SE = 0
        best_APassignment = np.zeros(M)

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R_small, nbrOfRealizations, M, K_small, N, tau_p, pilotIndex_small, p)

        # Try each AP assignment:
        for idx, APassignment in enumerate(feasible_APassignments):

            D_small[:, 0] = APassignment

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D_small, C, tau_c, tau_p,
                                                        nbrOfRealizations, N, K_small, M, p)

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

            SEs[idx] = sum_SE
            APcount = np.sum(APassignment == 1)

        best_SE = np.max(SEs)
        best_APassignment = feasible_APassignments[np.argmax(SEs)]


    pilotIndex = np.hstack((old_pilotIndex, [best_pilot]))

    assignAPs, = np.where(best_APassignment == 1)
    assingmColumn = np.zeros((L, 1))
    assingmColumn[bestAPsindex[assignAPs]] = 1
    D = np.hstack((old_D, assingmColumn))

    print('Best AP assigment: ', best_APassignment)
    print('Sum SE: ', best_SE)

    return pilotIndex, D, D_small[:, :-1], R_small, pilotIndex_small[:-1], best_pilot, best_APassignment



def AP_Pilot_newUE_Benchmarks(R, gainOverNoisedB, tau_p, L, N, old_pilotIndex, mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param ...
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    K = gainOverNoisedB.shape[1]

    # to store pilot assignment
    pilotIndex = np.hstack((old_pilotIndex, [-1]))

    # to store AP assignment
    D = np.zeros((L, K))

    # check for PA mode
    match mode:
        case 'random':
            print('implement random')

        case 'DCC':
            # Determine the master AP for UE k by looking for the AP with best channel condition
            master = np.argmax(gainOverNoisedB[:, -1])

            # Set the master AP as serving AP
            D[master, -1] = 1

            if K - 1 <= tau_p - 1:  # Assign orthogonal pilots to the first tau_p UEs
                pilotIndex[-1] = K - 1

            else:  # Assign pilot for remaining users

                # Compute received power to the master AP from each pilot
                pilotInterference = np.zeros(tau_p)

                for t in range(tau_p):
                    pilotInterference[t] = np.sum(db2pow(gainOverNoisedB[master, :-1][pilotIndex[:-1] == t]))

                # Find the pilot with least received power
                bestPilot = np.argmin(pilotInterference)
                pilotIndex[-1] = bestPilot

            # Guarantee that each UE served at least by the master AP
            for k in range(K - 1):
                # Determine the master AP for UE k by looking for the AP with best channel condition
                master = np.argmax(gainOverNoisedB[:, k])

                # Set the master AP as serving AP
                D[master, k] = 1

            # Each AP serves the UE with the strongest channel condition on each of the pilots
            for l in range(L):
                for t in range(tau_p):
                    pilotUEs, = np.where(pilotIndex == t)
                    if len(pilotUEs) > 0:
                        UEindex = np.argmax(gainOverNoisedB[l, pilotIndex == t])
                        D[l, pilotUEs[UEindex]] = 1

        case 'ALL':

            # Determine the master AP for UE k by looking for the AP with best channel condition
            master = np.argmax(gainOverNoisedB[:, -1])

            # Set the master AP as serving AP
            D[master, -1] = 1

            if K - 1 <= tau_p - 1:  # Assign orthogonal pilots to the first tau_p UEs
                pilotIndex[-1] = K - 1

            else:  # Assign pilot for remaining users

                # Compute received power to the master AP from each pilot
                pilotInterference = np.zeros(tau_p)

                for t in range(tau_p):
                    pilotInterference[t] = np.sum(db2pow(gainOverNoisedB[master, :-1][pilotIndex[:-1] == t]))

                # Find the pilot with least received power
                bestPilot = np.argmin(pilotInterference)
                pilotIndex[-1] = bestPilot

            D = np.ones((L, K))

    return pilotIndex, D
