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

def generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta, nbrOfSetups, pilot_alloc_mode, seed = 1):
    """Generates realizations of the setup
    INPUT>
    :param L: Number of APs per setup
    :param K: Number of UEs in the network
    :param N: Number of antennas per AP
    :param tau_p: Number of orthogonal pilots
    :param ASD_varphi: Angular standard deviation in the local scattering model
                       for the azimuth angle (in radians)
    :param ASD_theta: Angular standard deviation in the local scattering model
                       for the elevation angle (in radians)
    :param nbrOfSetups: Number of setups with random UE and AP locations
    :param pilot_alloc_mode: Pilot allocation mode
    :param seed: Seed number of pseudorandom number generator


    OUTPUT>
    gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    D: DCC matrix with dimensions LxK where the element (l,k) equals '1' if Ap l serves
                        UE k, and '0' otherwise
    D_small: DCC matrix with dimensions LxK where the element (l,k) equals '1' if Ap l serves
                        UE k, and '0' otherwise (for small-cell setups)
    APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    distances: matrix of dimensions LxK where element (l,k) is the distance en meters between
                      Ap l and UE k
    """
    print(f'Pilot allocation mode: {pilot_alloc_mode}')

    np.random.seed(seed)

    # Simulation Setup Configuration Parameters
    squarelength = 1000         # length of one side the coverage area in m (assuming wrap-around)

    B = 20*10**6                # communication bandwidth in Hz
    noiseFigure = 7             # noise figure in dB
    noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm

    alpha = 36.7                # pathloss parameters for the path loss model
    constantTerm = -30.5

    sigma_sf = 4                # standard deviation of the shadow fading
    decorr = 9                  # decorrelation distance of the shadow fading

    distanceVertical = 10       # height difference between the APs and the UEs in meters
    antennaSpacing = 0.5        # half-wavelength distance

    # To save the results
    gainOverNoisedB = np.zeros((L, K))
    R = np.zeros((N, N, L, K), dtype=complex)
    distances = np.zeros((L, K))
    pilotIndex = np.zeros((K))
    D = np.zeros((L, K))
    D_small = np.zeros((L, K))
    masterAPs = np.zeros((K, 1))        # stores the indices of the master AP of every UE
    APpositions = np.zeros((L, 1))
    UEpositions = np.zeros((K, 1), dtype=complex)

    # APpositions = (np.random.rand(L, 1) + 1j*np.random.rand(L, 1))*squarelength     # random AP locations with uniform distribution
    # for this test we use fixed centered AP
    APpositions = np.array([[500 + 500j]], dtype=complex)

    # To save the shadowing correlation matrices
    shadowCorrMatrix = sigma_sf**2*np.ones((K, K))
    shadowAPrealizations = np.zeros((K, L))

    # # fixed UE positions (for test)
    # _UEpositions = [200+200j, 204+208j, 500+800j, 700+800j, 100+100j]

    # Add UEs
    for k in range(K):
        # generate a random UE location with uniform distribution
        UEposition = (np.random.rand() + 1j*np.random.rand())*squarelength     # Uncomment when tests finish
        # UEposition = _UEpositions[k]            # Only for test


        # compute distance from new UE to all the APs
        distances[:, k] = np.sqrt(distanceVertical**2+np.abs(APpositions-UEposition)**2)[:, 0]

        if k > 0:         # if UE k is not the first UE
            shortestDistances = np.zeros((k, 1))

            for i in range(k):
                shortestDistances[i] = min(np.abs(UEposition-UEpositions[i]))

            # Compute conditional mean and standard deviation necessary to obtain the new shadow fading
            # realizations when the previous UEs' shadow fading realization have already been generated
            newcolumn = (sigma_sf**2)*(2**(shortestDistances/-(decorr)))[:, 0]
            term1 = newcolumn.conjugate().T@alg.inv(shadowCorrMatrix[:k, :k])
            meanvalues = term1@shadowAPrealizations[:k, :]
            stdvalue = np.sqrt(sigma_sf**2 - term1@newcolumn)

        else:           # if UE k is the first UE
            meanvalues = 0
            stdvalue = sigma_sf
            newcolumn = np.array([])

        shadowing = meanvalues + stdvalue*np.random.randn(L)   # generate the shadow fading realizations
                                                              # arreglar randn>rand

        # Compute the channel gain divided by noise power
        # gainOverNoisedB[:, k] = constantTerm - alpha * np.log10(distances[:, k]) - noiseVariancedBm
        # In this test we eliminated the random contribution to the gain (shadowing)
        gainOverNoisedB[:, k] = constantTerm - alpha * np.log10(distances[:, k]) + shadowing - noiseVariancedBm

        # Update shadowing correlation matrix and store realizations
        shadowCorrMatrix[0:k, k] = newcolumn
        shadowCorrMatrix[k, 0:k] = newcolumn.T
        shadowAPrealizations[k, :] = shadowing

        # store the UE position
        UEpositions[k] = UEposition

    # setup map
    drawPilotAssignment(UEpositions, APpositions, np.zeros(K, dtype=int), title="User allocation " + pilot_alloc_mode)


    # Compute correlation matrices
    for k in range(K):
        # run over the APs
        for l in range(L):  # Go through all APs
            angletoUE_varphi = np.angle(UEpositions[k] - APpositions[l])
            angletoUE_theta = np.arcsin(distanceVertical / distances[l, k])

            # Generate the approximate spatial correlation matrix using the local scattering model by scaling
            # the normalized matrices with the channel gain
            R[:, :, l, k] = db2pow(gainOverNoisedB[l, k]) * localScatteringR(N, angletoUE_varphi, ASD_varphi,
                                                                             antennaSpacing)
            # R[:, :, l, k] = localScatteringR(N, angletoUE_varphi, ASD_varphi, antennaSpacing)

    # Compute the pilot assignment. Modes = ['DCC', ]
    pilotIndex = pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, distances, gainOverNoisedB, R,
                                 mode=pilot_alloc_mode)

    # every user is served at least by its master AP
    for k in range(K):
        # Determine the master AP for UE k by looking for the AP with best channel condition
        master = np.argmax(gainOverNoisedB[:, k])

        # serve user k by its master
        D[master, k] = 1

    # # Each AP serves the UE with the strongest channel condition on each of the pilots
    # for l in range(L):
    #     for t in range(tau_p):
    #         pilotUEs, = np.where(pilotIndex == t)
    #         if len(pilotUEs) > 0:
    #             UEindex = np.argmax(gainOverNoisedB[l, pilotIndex == t])
    #             D[l, pilotUEs[UEindex]] = 1

    # # To draw the assignation of APs to the users
    # serving_APs = np.argmax(D, axis=0)
    # drawPilotAssignment(UEpositions, APpositions, serving_APs, title="User allocation "+pilot_alloc_mode)
    #
    # for k in range(K):
    #     for l in range(L):
    #         R[:, :, l, k] = db2pow(gainOverNoisedB[l, k]) * R[:, :, l, k]



    return gainOverNoisedB, R, pilotIndex, D, D_small

def pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, distances, gainOverNoisedB, R, mode):
    """return the pilot allocation
    INPUT>
    :param K: number of users
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

    # to store pilot assignment
    pilotIndex = np.zeros((K), int)

    # check for DCC mode
    if mode == 'DCC':

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

    elif mode == 'Kmeans_basic_positions':
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

        # # lowest SSE value
        # print("minimum SSE: {}".format(min_SSE := kmeans.inertia_))

        # Final locations of the centroid
        clusters_centroid = kmeans.cluster_centers_

        # # iterations to converge
        # print("Number of iterations to converge {}".format(iter_converge := kmeans.n_iter_))

        # get UE clustering
        UE_clustering = kmeans.labels_

        # draw UE clustering
        drawPilotAssignment(UEpositions, APpositions, UE_clustering, title='UE Clustering')

        # compute pilot allocation for the obtained clustering
        pilotIndex = inCluster_pilotAllocation(UE_clustering, R, gainOverNoisedB, K, tau_p, N, mode='best_first')

    elif mode == 'Kmeans_basic_R':
        kmeans = KMeans(init = "random",
            n_clusters = tau_p,
            n_init = 10,
            max_iter = 300,
            random_state = 42
                        )

        features = []
        features_eigenvectors = []
        for k in range(K):
            # Determine the master AP for UE k by looking for the AP with best channel condition
            master = np.argmax(gainOverNoisedB[:, k])

            # Get the correlation matrix associated with the master AP
            # R_k = R[:, :, master, k]/db2pow(gainOverNoisedB[master, k])
            R_k = R[:, :, master, k]

            # Compute the eigenvectors and get the eigenvector associated with largest eigenvalue
            eigenvalues, eigenvectors = np.linalg.eig(R_k)
            index = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, index]
            eigenvalues = eigenvalues[index]
            flatten_complex_features = eigenvectors[-1].flatten()

            features.append(np.concatenate((flatten_complex_features.real, flatten_complex_features.imag)))
            features_eigenvectors.append(flatten_complex_features)

        # scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)      # uncomment after tests

        # perform clustering
        kmeans.fit(scaled_features)

        # # lowest SSE value
        # print("minimum SSE: {}".format(min_SSE := kmeans.inertia_))

        # Final locations of the centroid
        clusters_centroid = kmeans.cluster_centers_

        # # iterations to converge
        # print("Number of iterations to converge {}".format(iter_converge := kmeans.n_iter_))

        # get UE clustering
        UE_clustering = kmeans.labels_

        # draw UE clustering
        drawPilotAssignment(UEpositions, APpositions, UE_clustering, title='UE Clustering '+mode)

        # compute pilot allocation for the obtained clustering
        pilotIndex = inCluster_pilotAllocation(UE_clustering, gainOverNoisedB, K, tau_p, L, mode='worst_first')

    elif mode == 'Kbeams':
        # number of initial beams
        number_beams = tau_p
        # compute equispaced angles for the initial beams
        initial_angles_deg = np.linspace(-85, 85, number_beams + 1)[:-1]
        initial_angles = np.radians(initial_angles_deg)

        # to store the correlation matrices for the initial beams
        R_beams = np.zeros((number_beams, N, N), dtype=complex)

        # compute the correlation matrices for the initial beams
        for beam in range(number_beams):
            R_b = localScatteringR(N, initial_angles[beam])
            R_beams[beam, :, :] = R_b / np.linalg.norm(R_b)

        # to store the correlation matrices for the UEs in a different structure
        R_UEs = np.zeros((K, N, N), dtype=complex)

        # re-store the R matrices for the UEs
        for k in range(K):
            for l in range(L):  # Go through all APs
                angle_k = np.angle(UEpositions[k] - APpositions[l])
                R_k = localScatteringR(N, angle_k)
                R_UEs[k, :, :] = R_k / np.linalg.norm(R_k)

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

        drawPilotAssignment(UEpositions, APpositions, pilotIndex, title='UE Clustering ')
        #

        updating = True
        while updating:
            # beam_allocation = new_beam_allocation
            new_beam_allocation = [[] for _ in range(number_beams)]
            correlation_factors = np.zeros((K, number_beams))
            for k in range(K):
                # correlation_factors = np.zeros(number_beams)
                for beam in range(number_beams):
                    correlation_factors[k, beam] = np.abs(
                        np.vdot(np.array(new_R_beams[beam, :, :]), np.array(R_UEs[k, :, :])))
                matching_beam = np.argmax(correlation_factors[k, :])
                new_beam_allocation[matching_beam].append(k)
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

            drawPilotAssignment(UEpositions, APpositions, UE_clustering, title='UE Clustering ')
            #

        # compute pilot allocation for the obtained clustering
        pilotIndex = inCluster_pilotAllocation(UE_clustering, R, gainOverNoisedB, K, tau_p, N, mode='best_first')




    drawPilotAssignment(UEpositions, APpositions, pilotIndex, title='Pilot Assignment '+mode)
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
            p = c
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

    return pilotIndex



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

def db2pow(dB):
    """return the power values that correspond to the input dB values
    INPUT>
    :param dB: dB input value
    OUTPUT>
    pow: power value
    """
    pow = 10**(dB/10)
    return pow

def localScatteringR(N, nominalAngle, ASD=math.radians(5), antennaSpacing=0.5):
    """return the approximate spatial correlation matrix for the local scattering model
    INPUT>
    :param N: number of antennas at the AP
    :param nominalAngle: nominal azimuth angle
    :param ASD: angular standard deviation around the nominal azimuth angle in radians

    OUTPUT>
    R: spatial correlation matrix
    """

    firstColumn = np.zeros((N), dtype=complex)

    for column in range(N):
        distance = column

        firstColumn[column] = np.exp(1j * 2 * np.pi * antennaSpacing * np.sin(nominalAngle) * distance) * np.exp(
            (-(ASD ** 2) / 2) * (2 * np.pi * antennaSpacing * np.cos(nominalAngle) * distance) ** 2)

    R = spalg.toeplitz(firstColumn)

    return np.matrix(R).T