import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from functionsUtils import db2pow, localScatteringR, correlationNormalized_grid

def pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, distances, gainOverNoisedB, R, **modes):
    """return the pilot allocation
    INPUT>
    :param K: number of users
    :param L: number of APs
    :param tau_p: Number of orthogonal pilots
    :param APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    :param UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    :param distances: matrix of dimensions LxK where element (l,k) is the distance en meters between
                      Ap l and UE k
    :param gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    :param mode: select the clustering mode
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    if len(modes) == 1:
        clustering_mode = modes['PA_mode']
        print(f'classic pilot allocation mode: {clustering_mode}')

    elif len(modes) == 2:
        clustering_mode = modes['clustering_mode']
        print(f'Clustering mode: {clustering_mode}')
        PA_mode = modes['PA_mode']
        print(f'PA mode: {PA_mode}')

    elif len(modes) == 3:
        clustering_mode = modes['clustering_mode']
        print(f'Clustering mode: {clustering_mode}')
        PA_mode = modes['PA_mode']
        print(f'PA mode: {PA_mode}')
        init_mode = modes['init_mode']
        print(f'Init mode: {init_mode}')

    # to store pilot assignment
    pilotIndex = np.zeros((K), int)

    # check for DCC mode
    if clustering_mode == 'DCC':

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

    elif clustering_mode == 'SGPS':
        unsched_users = set(np.arange(0, K))
        scheduled_users = set()
        pilotIndex = -1 * np.ones((K), int)

        # Assign pilot 0 to user 0
        pilotIndex[0] = 0
        unsched_users.remove(0)
        scheduled_users.add(0)

        # to store similarity
        similarity = np.zeros(K)

        # Assign pilots from 1 to tau_p to users more similar to users already allocated
        for t in range(1, tau_p):
            # Run over the users
            for k in unsched_users:
                similarity[k] = sum([(np.trace(R[:, :, 0, k]@R[:, :, 0, k_])
                                     /(np.linalg.norm(R[:, :, 0, k]) * np.linalg.norm(R[:, :, 0, k_])))
                for k_ in scheduled_users])
            similar_user = np.argmax(similarity)
            similarity[similar_user] = 0
            pilotIndex[similar_user] = t
            unsched_users.remove(similar_user)
            scheduled_users.add(similar_user)

        # Find the best pilot for every user
        interferenc_pilot = np.zeros(tau_p)

        # Run over the non-scheduled users
        for k in unsched_users:
            # Run over the pilots
            for t in range(tau_p):

                # Get the pilot-sharing users
                pilotsharing_users, = np.where(pilotIndex == t)

                interferenc_pilot[t] = sum([(np.trace(R[:, :, 0, k]@R[:, :, 0, k_])
                                     /(np.linalg.norm(R[:, :, 0, k]) * np.linalg.norm(R[:, :, 0, k_])))
                for k_ in pilotsharing_users])

            # get lower interference pilot and assign it to user k
            best_pilot = np.argmin(interferenc_pilot)
            pilotIndex[k] = best_pilot

    elif clustering_mode == 'Kmeans_basic_positions':
        kmeans = KMeans(init = "random",
            n_clusters = tau_p,
            n_init = 10,
            max_iter = 300,
            random_state = 42
                        )
        # Convert into a feature matrix with column0 => x_position and column1 => y_position
        features = np.concatenate((UEpositions.real, UEpositions.imag), axis=1)

        # scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # perform clustering
        kmeans.fit(scaled_features)

        # get UE clustering
        UE_clustering = kmeans.labels_

        # draw UE clustering
        # drawPilotAssignment(UEpositions, APpositions, UE_clustering, title='UE Clustering')

        # compute pilot allocation for the obtained clustering
        pilotIndex = inCluster_pilotAllocation(UE_clustering, R, gainOverNoisedB, K, tau_p, N, mode=PA_mode)

    elif clustering_mode == 'Kbeams':

        R_UEs, R_beams, number_beams = Kbeams_init(K, L, tau_p, N, UEpositions, APpositions, mode=init_mode)

        # to store the beam clustering
        beam_allocation = [[] for _ in range(number_beams)]

        correlation_factors = np.zeros((K, number_beams))
        for k in range(K):
            # correlation_factors = np.zeros(number_beams)
            for beam in range(number_beams):
                correlation_factors[k, beam] = np.abs(np.vdot(np.array(R_beams[beam, :, :]), np.array(R_UEs[k, :, :])))
            matching_beam = np.argmax(correlation_factors[k, :])
            beam_allocation[matching_beam].append(k)

        new_R_beams = []
        for beam in range(number_beams):
            if beam_allocation[beam]:
                new_R_beams.append(sum([R_UEs[allocated_user, :, :] for allocated_user in beam_allocation[beam]]))
        number_beams = len(new_R_beams)
        new_R_beams = np.array(new_R_beams)

        beam_allocation = list(filter(None, beam_allocation))

        # to draw
        UE_clustering = np.zeros((K), int)
        for beam in range(number_beams):
            for user in beam_allocation[beam]:
                UE_clustering[user] = beam

        # drawPilotAssignment(UEpositions, APpositions, pilotIndex, title='UE Clustering ')
        #

        updating = True
        iter = 0
        while updating and iter<10:
            # beam_allocation = new_beam_allocation
            new_beam_allocation = [[] for _ in range(number_beams)]
            correlation_factors = np.zeros((K, number_beams))
            for k in range(K):
                # correlation_factors = np.zeros(number_beams)
                for beam in range(number_beams):
                    if k in beam_allocation[beam]:
                        correlation_factors[k, beam] = np.abs(
                            np.vdot(np.array(new_R_beams[beam, :, :]-R_UEs[k, :, :]), np.array(R_UEs[k, :, :])))
                    else:
                        correlation_factors[k, beam] = np.abs(
                            np.vdot(np.array(new_R_beams[beam, :, :]), np.array(R_UEs[k, :, :])))

                matching_beam = np.argmax(correlation_factors[k, :])
                new_beam_allocation[matching_beam].append(k)
            iter += 1
            new_R_beams = []
            for beam in range(number_beams):
                if new_beam_allocation[beam]:
                    new_R_beams.append(
                        sum([R_UEs[allocated_user, :, :] for allocated_user in new_beam_allocation[beam]]))
            number_beams = len(new_R_beams)
            new_R_beams = np.array(new_R_beams)

            new_beam_allocation = list(filter(None, new_beam_allocation))

            if new_beam_allocation == beam_allocation:
                updating = False
            else:
                beam_allocation = new_beam_allocation

            # to draw
            UE_clustering = np.zeros((K), int)
            for beam in range(number_beams):
                for user in new_beam_allocation[beam]:
                    UE_clustering[user] = beam

            # drawPilotAssignment(UEpositions, APpositions, UE_clustering, title='UE Clustering ')
            #

        # compute pilot allocation for the obtained clustering
        pilotIndex = inCluster_pilotAllocation(UE_clustering, R, gainOverNoisedB, K, tau_p, N, mode=PA_mode)




    # drawPilotAssignment(UEpositions, APpositions, pilotIndex, title='Pilot Assignment '+clustering_mode)
    return pilotIndex

def inCluster_pilotAllocation(clustering, R, gainOverNoisedB, K, tau_p, N, mode):
    """Use clustering information to assign pilots to the UEs. UEs in the same cluster should be assigned
    different pilots
    INPUT>
    :param clustering: vector whose entry clustering[k] contains the index of cluster which UE k is assigned
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """
    pilotIndex = -1*np.ones((K), int)
    if mode=='basic':
        for c in range(max(clustering)+1):
            p = c
            usersInCluster, = np.where(clustering == c)
            for userInCluster in usersInCluster:
                pilotIndex[userInCluster] = p%tau_p
                p+=1
    elif mode=='worst_first':
        for c in range(max(clustering) + 1):
            p = c
            usersInCluster, = np.where(clustering == c)
            ordered_usersInCluster = sorted(usersInCluster, key=lambda x: gainOverNoisedB[0, x])
            for ind, user in enumerate(ordered_usersInCluster):
                if ind < tau_p:
                    pilotIndex[user] = p%tau_p
                    p+=1
                else:
                    interference = np.zeros(tau_p)
                    for t in range(tau_p):
                        interference[t] = sum(db2pow(gainOverNoisedB[0, usersInCluster]
                                                     [pilotIndex[usersInCluster]==t]))
                    pilotIndex[user] = np.argmin(interference)
    elif mode=='best_first':
        for c in range(max(clustering) + 1):
            p = c
            usersInCluster, = np.where(clustering == c)
            ordered_usersInCluster = sorted(usersInCluster, key=lambda x: gainOverNoisedB[0, x], reverse=True)
            for ind, user in enumerate(ordered_usersInCluster):
                if ind < tau_p:
                    pilotIndex[user] = p%tau_p
                    p+=1
                else:
                    interference = np.zeros(tau_p)
                    for t in range(tau_p):
                        interference[t] = sum(db2pow(gainOverNoisedB[0, usersInCluster]
                                                     [pilotIndex[usersInCluster]==t]))
                    pilotIndex[user] = np.argmin(interference)

    elif mode=='bf_NMSE':
        for c in range(max(clustering) + 1):
            p = 0
            usersInCluster, = np.where(clustering == c)
            ordered_usersInCluster = sorted(usersInCluster, key=lambda x: gainOverNoisedB[0, x], reverse=True)
            for ind, user in enumerate(ordered_usersInCluster):
                if ind < tau_p:
                    pilotIndex[user] = p%tau_p
                    p+=1
                else:
                    NMSE = np.zeros(tau_p)
                    for t in range(tau_p):
                        pilotSharing_UEs, = np.where(pilotIndex[usersInCluster] == t)
                        interference = alg.inv(np.identity(N)
                                               + sum([tau_p*p*R[:, :, 0, k] for k in usersInCluster[pilotSharing_UEs]])
                                               + tau_p*p*R[:, :, 0, user])
                        NMSE[t] = 1 - (tau_p*p*np.trace(R[:, :, 0, user]@interference@R[:, :, 0, user])/
                                np.trace(R[:, :, 0, user])).real
                    pilotIndex[user] = np.argmin(NMSE)

    elif mode=='wf_NMSE':
        for c in range(max(clustering) + 1):
            p = 0
            usersInCluster, = np.where(clustering == c)
            ordered_usersInCluster = sorted(usersInCluster, key=lambda x: gainOverNoisedB[0, x])
            for ind, user in enumerate(ordered_usersInCluster):
                if ind < tau_p:
                    pilotIndex[user] = p%tau_p
                    p+=1
                else:
                    sum_NMSE = np.zeros(tau_p)
                    for t in range(tau_p):
                        pilotSharing_UEs, = np.where(pilotIndex[usersInCluster] == t)
                        interference = alg.inv(np.identity(N)
                                               + sum(
                            [tau_p * p * R[:, :, 0, k] for k in usersInCluster[pilotSharing_UEs]])
                                               + tau_p * p * R[:, :, 0, user])
                        sum_NMSE[t] = sum([1 - (tau_p*p*np.trace(R[:, :, 0, k]@interference@R[:, :, 0, k])/
                                np.trace(R[:, :, 0, k])).real for k in usersInCluster[pilotSharing_UEs]])
                    pilotIndex[user] = np.argmin(sum_NMSE)
    return pilotIndex

def Kbeams_init(K, L, tau_p, N, UEpositions, APpositions, mode):
    '''Create the initial beams for the K-beam algorithm
    INPUT>
    :param K: number of users
    :param L: number of APs
    :param tau_p: Number of orthogonal pilots
    :param N: number of antennas at the BS
    :param UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    :param APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    :param mode: select the beam initialization mode
    OUTPUT>
    R_UEs: matrix of dimensions KxNxN that contains the normalized correlation matrices of users
    R_beams: matrix of dimensions number_of_beamsxNxN that contains the normalized correlation matrices of beams
    number_beams: number of initial beams
    '''

    # to store the correlation matrices for the UEs in a different structure
    R_UEs = np.zeros((K, N, N), dtype=complex)

    # get the angles from all the UEs to the AP 0 (in radians)
    angles_UEs = np.angle(UEpositions - APpositions[0])

    # re-store the R matrices for the UEs
    for k in range(K):
        for l in range(L):  # Go through all APs
            R_k = localScatteringR(N, angles_UEs[k])
            R_UEs[k, :, :] = R_k / np.linalg.norm(R_k)

    match mode:
        case 'basic_angle_spread':
            # number of initial beams
            number_beams = int(tau_p/1)

            # compute equispaced angles for the initial beams
            initial_angles_deg = np.linspace(-90, 90, number_beams + 1)[:-1]
            initial_angles = np.radians(initial_angles_deg)

            # to store the correlation matrices for the initial beams
            R_beams = np.zeros((number_beams, N, N), dtype=complex)

            # compute the correlation matrices for the initial beams
            for beam in range(number_beams):
                R_b = localScatteringR(N, initial_angles[beam])
                R_beams[beam, :, :] = R_b / np.linalg.norm(R_b)

        case 'dissimilar_K':
            # number of initial beams
            number_beams = tau_p

            first_user = np.random.randint(0, K - 1)

            initial_users = [first_user]

            for _ in range(number_beams-1):
                R_initial = np.array(sum([R_UEs[k, :, :] for k in initial_users]), dtype=complex)
                similarity_matrix = (R_UEs.reshape(K, -1) @ R_initial.reshape(-1, 1).conjugate()).real
                initial_users.append(np.argmin(similarity_matrix))

            # to store the correlation matrices for the initial beams
            R_beams = np.zeros((number_beams, N, N), dtype=complex)

            for idx, user in enumerate(initial_users):
                R_b = localScatteringR(N, angles_UEs[user])
                R_beams[idx, :, :] = R_b / np.linalg.norm(R_b)


    print('testing')

    return R_UEs, R_beams, number_beams


def drawPilotAssignment(UEpositions, APpositions, pilotIndex, title):
    """
    INPUT>
    :param UEpositions: (see above)
    :param pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    OUTPUT>
    """

    # create a custom color palette for up to 10 orthogonal pilots
    custom_colors = np.array(['red', 'dodgerblue', 'green', 'orchid', 'aqua', 'orange', 'lime', 'black', 'pink', 'yellow']*10)

    # pilot assignment graph
    plt.scatter(UEpositions.real, UEpositions.imag, c=custom_colors[pilotIndex], marker='*')
    plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^')
    plt.title(title)
    for i, txt in enumerate(range(len(UEpositions))):
        plt.annotate(txt, (UEpositions[i].real, UEpositions[i].imag))
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.show()


