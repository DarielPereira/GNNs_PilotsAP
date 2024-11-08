import numpy as np
import numpy.linalg as alg


from functionsUtils import db2pow, localScatteringR, drawingSetup, drawing3Dvectors
# from functionsClustering_PilotAlloc import pilotAssignment, drawPilotAssignment


def generateSetup(L, K, N, cell_side, ASD_varphi, seed = 0):
    """Generates realizations of the setup
    INPUT>
    :param L: number of APs
    :param K: Number of UEs in the network
    :param N: Number of antennas per AP
    :param cell_side: Cell side length
    :param ASD_varphi: Angular standard deviation in the local scattering model
                       for the azimuth angle (in radians)
    :param seed: Seed number of pseudorandom number generator


    OUTPUT>
    gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
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
    squarelength = cell_side    # length of one side the coverage area in m

    B = 20*10**6                # communication bandwidth in Hz
    noiseFigure = 7             # noise figure in dB
    noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm

    alpha = 36.7                # pathloss parameters for the path loss model
    constantTerm = -30.5

    distanceVertical = 10       # height difference between the APs and the UEs in meters
    antennaSpacing = 0.5        # half-wavelength distance

    AP_spacing = squarelength/(L**0.5)

    # Regular AP deployment
    APpositions = []
    for i in np.linspace(AP_spacing/2, squarelength - AP_spacing/2, int(L**0.5)):
        for j in np.linspace(AP_spacing/2, squarelength - AP_spacing/2, int(L**0.5)):
            APpositions.append(complex(i, j) + complex(20*np.random.randn(1)[0], 20*np.random.randn(1)[0]))
    APpositions = np.array(APpositions).reshape(-1, 1)

    # To save the results
    gainOverNoisedB = np.zeros((L, K))
    R = np.zeros((N, N, L, K), dtype=complex)
    distances = np.zeros((L, K))
    UEpositions = np.zeros((K, 1), dtype=complex)


    # Add UEs
    for k in range(K):
        # generate a random UE location with uniform distribution
        UEposition = (np.random.rand() + 1j*np.random.rand())*squarelength     # Uncomment when tests finish

        # compute distance from new UE to all the APs
        distances[:, k] = np.sqrt(distanceVertical**2+np.abs(APpositions-UEposition)**2)[:, 0]

        # Compute the channel gain divided by noise power
        gainOverNoisedB[:, k] = constantTerm - alpha * np.log10(distances[:, k]) - noiseVariancedBm

        # store the UE position
        UEpositions[k] = UEposition

    # setup map
    # drawingSetup(UEpositions, APpositions, np.zeros((K), dtype=int), title="Setup Map", squarelength=squarelength)


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

    return gainOverNoisedB, distances, R, APpositions, UEpositions


def insertNewUE(L, N, cell_side, ASD_varphi, old_gainOverNoisedB, old_distances, old_R, APpositions, old_UEpositions, seed = 1000):
    """Generates realizations of the setup
    INPUT>
    :param L: number of APs
    :param K: Number of UEs in the network
    :param N: Number of antennas per AP
    :param cell_side: Cell side length
    :param ASD_varphi: Angular standard deviation in the local scattering model
                       for the azimuth angle (in radians)
    :param seed: Seed number of pseudorandom number generator


    OUTPUT>
    gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
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
    squarelength = cell_side   # length of one side the coverage area in m

    B = 20*10**6                # communication bandwidth in Hz
    noiseFigure = 7             # noise figure in dB
    noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm

    alpha = 36.7                # pathloss parameters for the path loss model
    constantTerm = -30.5

    distanceVertical = 10       # height difference between the APs and the UEs in meters
    antennaSpacing = 0.5        # half-wavelength distance

    # To save the results
    gainOverNoisedB = np.hstack((old_gainOverNoisedB, np.zeros((L, 1))))
    R = np.zeros((N, N, L, old_R.shape[3]+1), dtype=complex)
    for k in range(old_R.shape[3]):
        R[:, :, :, k] = np.array(old_R[:, :, :, k])

    distances = np.hstack((old_distances, np.zeros((L, 1))))
    UEpositions = np.zeros((old_UEpositions.shape[0]+1, 1), dtype=complex)
    UEpositions[0:-1, 0] = old_UEpositions[:, 0]


    UEpositions[-1] = (np.random.rand() + 1j*np.random.rand())*squarelength

    distances[:, -1] = np.sqrt(distanceVertical**2+np.abs(APpositions-UEpositions[-1])**2)[:, 0]

    gainOverNoisedB[:, -1] = constantTerm - alpha * np.log10(distances[:, -1]) - noiseVariancedBm


    # run over the APs
    for l in range(L):  # Go through all APs
        angletoUE_varphi = np.angle(UEpositions[-1] - APpositions[l])

        # Generate the approximate spatial correlation matrix using the local scattering model by scaling
        # the normalized matrices with the channel gain
        R[:, :, l, -1] = db2pow(gainOverNoisedB[l, -1]) * localScatteringR(N, angletoUE_varphi, ASD_varphi,
                                                                           antennaSpacing)

    return gainOverNoisedB, distances, R, APpositions, UEpositions











def generateSetup_TesT(L, K, N, tau_p, ASD_varphi, ASD_theta, nbrOfRealizations, seed = 0, mode='test'):
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
    APpositions = np.array([[200 + 200j], [200 + 700j], [700 + 500j]], dtype=complex)

    # To save the shadowing correlation matrices
    shadowCorrMatrix = sigma_sf**2*np.ones((K, K))
    shadowAPrealizations = np.zeros((K, L))

    # # fixed UE positions (for test)
    _UEpositions = [250+240j, 264+288j, 150+250j, 260 + 710j, 185 + 750j, 150 + 650j, 275 + 760j, 605 + 570j,
                    655 + 540j, 772 + 542j, 400+200j, 500+300j]

    # Add UEs
    for k in range(K):
        # generate a random UE location with uniform distribution
        # UEposition = (np.random.rand() + 1j*np.random.rand())*squarelength     # Uncomment when tests finish
        UEposition = _UEpositions[k]            # Only for test


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
        gainOverNoisedB[:, k] = constantTerm - alpha * np.log10(distances[:, k]) - noiseVariancedBm   # In this test
        # we eliminated the random contribution to the gain (shadowing)
        # gainOverNoisedB[:, k] = constantTerm - alpha * np.log10(distances[:, k]) + shadowing - noiseVariancedBm

        # Update shadowing correlation matrix and store realizations
        shadowCorrMatrix[0:k, k] = newcolumn
        shadowCorrMatrix[k, 0:k] = newcolumn.T
        shadowAPrealizations[k, :] = shadowing

        # store the UE position
        UEpositions[k] = UEposition

    # setup map
    # drawingSetup(UEpositions, APpositions, np.array([0,0,0,1,1,1,1,2,2,2,3,3]), title="Setup Map", squarelength=squarelength)


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





    return gainOverNoisedB, R, APpositions, UEpositions
