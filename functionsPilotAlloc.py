import numpy as np
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

def pilotAssignment(clustering, R, gainOverNoisedB, K, tau_p, L, N, mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param clustering: vector whose entry clustering[k] contains the index of cluster which UE k is assigned
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """


    # to store pilot assignment
    pilotIndex = -1*np.ones((K), int)

    # check for PA mode
    match mode:
        case 'random':
            # print('PA mode: random')
            pilotIndex = np.array(random.choices(range(tau_p), k=K))

        case 'balanced_random':
            # print('PA mode: balanced random')
            for k in range(K):
                pilotIndex[k] = k % tau_p

        case 'DCC':

            # Determine the pilot assignment
            for k in range(K):

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

        case 'DCPA':
            # print('PA mode: DCPA')

            orthogonality = np.zeros((K))
            gainOverNoise = db2pow(gainOverNoisedB)

            ref = np.sum(gainOverNoisedB, axis=1)

            for k in range(K):
                orthogonality[k] = (np.abs(np.vdot(np.array(gainOverNoise[:, k]), np.array(ref))) /
                                   (linalg.norm(np.array(gainOverNoise[:, k]))*linalg.norm(np.array(ref))))
            sorted_indices = np.argsort(orthogonality)

            for k in range(K):
                pilotIndex[sorted_indices[k]] = k % tau_p

        case 'basic_iC':
            # print('PA mode: basic_iC')
            for c in range(max(clustering)+1):
                p = c
                usersInCluster, = np.where(clustering == c)
                for userInCluster in usersInCluster:
                    pilotIndex[userInCluster] = p % tau_p
                    p += 1
        case 'bf_iC':
            # print('PA mode: bf_iC')
            for c in range(max(clustering) + 1):
                p = c
                usersInCluster, = np.where(clustering == c)
                ordered_usersInCluster = sorted(usersInCluster, key=lambda x: max(gainOverNoisedB[:, x]), reverse=True)
                for ind, user in enumerate(ordered_usersInCluster):
                    if ind < tau_p:
                        pilotIndex[user] = p % tau_p
                        p += 1
                    else:
                        interference = np.zeros(tau_p)
                        for t in range(tau_p):
                            interference[t] = sum(db2pow(gainOverNoisedB[0, usersInCluster]
                                                         [pilotIndex[usersInCluster] == t]))
                        pilotIndex[user] = np.argmin(interference)

        case 'bfNMSE_iC':
            for c in range(max(clustering) + 1):
                p = c
                usersInCluster, = np.where(clustering == c)
                ordered_usersInCluster = sorted(usersInCluster, key=lambda x: max(gainOverNoisedB[:, x]), reverse=True)
                for ind, user in enumerate(ordered_usersInCluster):
                    if ind < tau_p:
                        pilotIndex[user] = p % tau_p
                        p += 1
                    else:
                        bestAP = np.argmax(gainOverNoisedB[:, user])
                        NMSE = np.zeros(tau_p)
                        for t in range(tau_p):
                            pilotSharing_UEs, = np.where(pilotIndex[usersInCluster] == t)
                            interference = linalg.inv(np.identity(N)
                                                   + sum([tau_p*p*R[:, :, bestAP, k] for k in usersInCluster[pilotSharing_UEs]])
                                                   + tau_p*p*R[:, :, bestAP, user])
                            NMSE[t] = 1 - (tau_p*p*np.trace(R[:, :, bestAP, user]@interference@R[:, :, bestAP, user])/
                                    np.trace(R[:, :, bestAP, user])).real
                        pilotIndex[user] = np.argmin(NMSE)

        case 'bf_master_iC':
            masters = np.argmax(gainOverNoisedB, axis=0)
            for c in range(max(clustering) + 1):
                p = 0
                usersInCluster, = np.where(clustering == c)
                ordered_usersInCluster = sorted(usersInCluster, key=lambda x: max(gainOverNoisedB[:, x]), reverse=True)
                for ind, user in enumerate(ordered_usersInCluster):
                    if ind < tau_p:
                        pilotIndex[user] = p % tau_p
                        p += 1
                    else:
                        commonMatser_UEs, = np.where(masters == masters[user])
                        bestAP = np.argmax(gainOverNoisedB[:, user])
                        NMSE = np.zeros(tau_p)
                        for t in range(tau_p):
                            pilotSharing_UEs, = np.where(pilotIndex[usersInCluster] == t)
                            potential_interf_UEs = np.array(list(set(commonMatser_UEs).intersection(set(pilotSharing_UEs))))
                            if len(potential_interf_UEs) != 0:
                                interference = linalg.inv(np.identity(N)
                                                          + sum(
                                    [tau_p * p * R[:, :, bestAP, k] for k in usersInCluster[potential_interf_UEs]])
                                                          + tau_p * p * R[:, :, bestAP, user])
                                NMSE[t] = 1 - (tau_p * p * np.trace(
                                    R[:, :, bestAP, user] @ interference @ R[:, :, bestAP, user]) /
                                               np.trace(R[:, :, bestAP, user])).real
                            else:
                                NMSE[t] = 0
                        pilotIndex[user] = np.argmin(NMSE)

        case 'bf_bAPs_iC':
            for c in range(max(clustering) + 1):
                p = c
                usersInCluster, = np.where(clustering == c)
                ordered_usersInCluster = sorted(usersInCluster, key=lambda x: max(gainOverNoisedB[:, x]), reverse=True)
                for ind, user in enumerate(ordered_usersInCluster):
                    if ind < tau_p:
                        pilotIndex[user] = p % tau_p
                        p += 1
                    else:
                        bestAPs = gainOverNoisedB[:, user].argsort()[-3:]
                        NMSE = np.zeros(tau_p)
                        interference = np.zeros((N, N, 3), dtype=complex)
                        for t in range(tau_p):
                            pilotSharing_UEs, = np.where(pilotIndex[usersInCluster] == t)
                            for idx, l in enumerate(bestAPs):
                                interference[:, :, idx] = linalg.inv(np.identity(N)
                                                      + sum(
                                [tau_p * p * R[:, :, l, k] for k in usersInCluster[pilotSharing_UEs]])
                                                      + tau_p * p * R[:, :, l, user])
                            NMSE[t] = 1 - (sum([tau_p * p * np.trace(
                                R[:, :, l, user] @ interference[:, :, idx] @ R[:, :, l, user]) for idx,l in enumerate(bestAPs)]) /
                                           sum([np.trace(R[:, :, l, user]) for l in bestAPs]).real)
                        pilotIndex[user] = np.argmin(NMSE)



    # selecting the serving APs
    # D = np.ones((L, K))
    # every user is served at least by its master AP
    D = np.zeros((L, K))
    for k in range(K):
        # Determine the master AP for UE k by looking for the AP with best channel condition
        master = np.argmax(gainOverNoisedB[:, k])
        # serve user k by its master
        D[master, k] = 1

    # Each AP serves the UE with the strongest channel condition on each of the pilots
    for l in range(L):
        for t in range(tau_p):
            pilotUEs, = np.where(pilotIndex == t)
            if len(pilotUEs) > 0:
                UEindex = np.argmax(gainOverNoisedB[l, pilotIndex == t])
                D[l, pilotUEs[UEindex]] = 1

    return pilotIndex, D


