import numpy as np
import numpy.linalg as alg

from functionsUtils import db2pow, localScatteringR
from functionsClustering_PilotAlloc import pilotAssignment, drawPilotAssignment

def generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta, nbrOfRealizations, seed = 2, **modes):
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
    :param nbrOfRealizations: Number of realizations with random UE and AP locations
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
    pilotIndex = np.zeros((K), dtype=int)
    clustering = np.zeros((K), dtype=int)
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
    # drawPilotAssignment(UEpositions, APpositions, np.zeros(K, dtype=int), title="Setup Map")


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
    if modes['clustering_mode'] == 'OPA':
        print('mode: OPA')
        pilotIndex = np.array(np.arange(K), dtype=int)

        # every user is served at least by its master AP
        for k in range(K):
            # all the UEs served by the only AP
            D[0, k] = 1

    elif modes['clustering_mode'] == 'OPA_LU':
        print('mode: OPA_LU')
        sorted_indexes = gainOverNoisedB[0].argsort()[::-1][:tau_p]

        for k in list(sorted_indexes):
            # only these UEs are served
            D[0, k] = 1

        pilotIndex = pilotIndex - 1
        pilotIndex[sorted_indexes] = np.array(np.arange(tau_p, dtype=int))


    else:
        print('mode: '+modes['clustering_mode'])
        # Compute the pilot assignment.
        pilotIndex, clustering = pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, distances, gainOverNoisedB, R,
                                     **modes)

        # every user is served at least by its master AP
        for k in range(K):
            # Determine the master AP for UE k by looking for the AP with best channel condition
            master = np.argmax(gainOverNoisedB[:, k])

            # serve user k by its master
            D[master, k] = 1

    return gainOverNoisedB, R, pilotIndex, clustering, D, D_small


def generateSetup_wUncorrelatedUEs(L, K, N, tau_p, ratioUncorrelated, ASD_varphi, ASD_theta, nbrOfRealizations, seed = 2, **modes):
    """Generates realizations of the setup
    INPUT>
    :param L: Number of APs per setup
    :param K: Number of UEs in the network
    :param N: Number of antennas per AP
    :param tau_p: Number of orthogonal pilots
    :param ratioUncorrelated: Percentage of UEs with uncorrelated channels
    :param ASD_varphi: Angular standard deviation in the local scattering model
                       for the azimuth angle (in radians)
    :param ASD_theta: Angular standard deviation in the local scattering model
                       for the elevation angle (in radians)
    :param nbrOfRealizations: Number of realizations with random UE and AP locations
    :param pilot_alloc_mode: Pilot allocation mode
    :param seed: Seed number of pseudorandom number generator
    :param modes: Dictionary of modes


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
    pilotIndex = np.zeros((K), dtype=int)
    clustering = np.zeros((K), dtype=int)
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
    # drawPilotAssignment(UEpositions, APpositions, np.zeros(K, dtype=int), title="Setup Map")


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

    # convert ratioUncorrelated*K UEs to uncorrelated channels
    number_uncorrelated = round(ratioUncorrelated*K)
    uncorrelatedUEs = np.random.choice(np.arange(K), number_uncorrelated, replace=False)

    for k in list(uncorrelatedUEs):
        for l in range(L):
            R[:, :, l, k] = db2pow(gainOverNoisedB[l, k]) * np.identity(N)

    if modes['clustering_mode'] == 'OPA':
        print('mode: OPA')
        pilotIndex = np.array(np.arange(K), dtype=int)

        # every user is served at least by its master AP
        for k in range(K):
            # all the UEs served by the only AP
            D[0, k] = 1

    elif modes['clustering_mode'] == 'OPA_LU':
        print('mode: OPA_LU')
        sorted_indexes = gainOverNoisedB[0].argsort()[::-1][:tau_p]

        for k in list(sorted_indexes):
            # only these UEs are served
            D[0, k] = 1

        pilotIndex = pilotIndex - 1
        pilotIndex[sorted_indexes] = np.array(np.arange(tau_p, dtype=int))


    else:
        print('mode: '+modes['clustering_mode'])
        # Compute the pilot assignment.
        pilotIndex, clustering = pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, distances, gainOverNoisedB, R,
                                     **modes)

        # every user is served at least by its master AP
        for k in range(K):
            # Determine the master AP for UE k by looking for the AP with best channel condition
            master = np.argmax(gainOverNoisedB[:, k])

            # serve user k by its master
            D[master, k] = 1

    return gainOverNoisedB, R, pilotIndex, clustering, D, D_small
