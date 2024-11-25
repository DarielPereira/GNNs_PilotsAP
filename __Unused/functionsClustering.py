import numpy as np
import numpy.linalg as linalg

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from functionsUtils import db2pow, drawingSetup

def cfMIMO_clustering(gainOverNoisedB, R, tau_p, APpositions, UEpositions, mode):
    """
    INPUT>
    :param gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                                (normalized by noise variance) between AP l and UE k
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                                matrix between  AP l and UE k (normalized by noise variance)
    :param tau_p: Number of orthogonal pilots
    :param APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                            where the real part is the horizontal position and the imaginary part is the
                            vertical position
    :param UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                            where the real part is the horizontal position and the imaginary part is the
                            vertical position
    OUTPUT>
    UE_clustering: matrix of dimensions Kx1 containing the UEs' clustering assignments
    """
    L = R.shape[2]
    K = R.shape[3]
    N = R.shape[0]

    # To store clustering
    UE_clustering = -1*np.ones(K, int)

    match mode:
        case 'Kfootprints':
            # print('clustering mode: K-footprints')

            # get the matrices with pathlosses in Power units
            gainMatrix = db2pow(gainOverNoisedB.T)

            R_UEs, R_footprints, number_footprints = Kfootprint_init(K, L, tau_p, N, R, UEpositions, APpositions, mode='disimilarUEs')

            # to store the footprint clustering
            footprint_allocation = [[] for _ in range(number_footprints)]

            correlation_factors = np.zeros((K, number_footprints))
            for k in range(K):
                # correlation_factors = np.zeros(number_footprints)
                for footprint in range(number_footprints):
                    correlation_factors[k, footprint] = np.abs(np.vdot(np.array(R_footprints[footprint, :, :]), np.array(R_UEs[k, :, :])))
                matching_footprint = np.argmax(correlation_factors[k, :])
                footprint_allocation[matching_footprint].append(k)
                # R_footprints[matching_footprint, :, :] = R_footprints[matching_footprint, :, :] + R_UEs[k, :, :]

            new_R_footprints = []
            for footprint in range(number_footprints):
                if footprint_allocation[footprint]:
                    new_R_footprints.append(sum([R_UEs[allocated_user, :, :] for allocated_user in footprint_allocation[footprint]]))
            number_footprints = len(new_R_footprints)
            new_R_footprints = np.array(new_R_footprints)

            footprint_allocation = list(filter(None, footprint_allocation))

            # to draw
            for footprint in range(number_footprints):
                for user in footprint_allocation[footprint]:
                    UE_clustering[user] = footprint

            # drawing initial clustering
            # drawingSetup(UEpositions, APpositions, UE_clustering, title='K-footprints initial clustering ', squarelength=1000)

            updating = True
            iter = 0
            while updating and iter < 10:
                new_footprint_allocation = [[] for _ in range(number_footprints)]
                correlation_factors = np.zeros((K, number_footprints))
                for k in range(K):
                    for footprint in range(number_footprints):
                        if k in footprint_allocation[footprint]:
                            new_R_footprints[footprint, :, :] = new_R_footprints[footprint, :, :] - R_UEs[k, :, :]
                            correlation_factors[k, footprint] = np.abs(
                                np.vdot(np.array(new_R_footprints[footprint, :, :]), np.array(R_UEs[k, :, :])))
                        else:
                            correlation_factors[k, footprint] = np.abs(
                                np.vdot(np.array(new_R_footprints[footprint, :, :]), np.array(R_UEs[k, :, :])))
                    matching_footprint = np.argmax(correlation_factors[k, :])
                    new_R_footprints[matching_footprint, :, :] = (
                            new_R_footprints[matching_footprint, :, :] + R_UEs[k, :, :])
                    new_footprint_allocation[matching_footprint].append(k)

                iter += 1
                new_R_footprints = []
                for footprint in range(number_footprints):
                    if new_footprint_allocation[footprint]:
                        new_R_footprints.append(
                            sum([R_UEs[allocated_user, :, :] for allocated_user in new_footprint_allocation[footprint]]))
                number_footprints = len(new_R_footprints)
                new_R_footprints = np.array(new_R_footprints)

                new_footprint_allocation = list(filter(None, new_footprint_allocation))

                if new_footprint_allocation == footprint_allocation:
                    updating = False
                else:
                    footprint_allocation = new_footprint_allocation

                for footprint in range(number_footprints):
                    for user in new_footprint_allocation[footprint]:
                        UE_clustering[user] = footprint

                # drawing clustering iterations
                # drawingSetup(UEpositions, APpositions, UE_clustering, title='UE Clustering ', squarelength=1000)

            # drawing final clustering
            # drawingSetup(UEpositions, APpositions, UE_clustering, title='K-footprints final clustering', squarelength=1000)

        case 'Kmeans_locations':
            # print('clustering mode: K-means locations')

            kmeans = KMeans(init="random",
                            # n_clusters=tau_p,
                            n_clusters=int(K / tau_p),
                            n_init=10,
                            max_iter=300,
                            random_state=42
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

            # drawing UE clustering
            # drawingSetup(UEpositions, APpositions, UE_clustering, title='K-means final clustering', squarelength=1000)

        case 'no_clustering':
            # print('clustering mode: no clustering')
            return UE_clustering

    return UE_clustering
        
        
        

def Kfootprint_init(K, L, tau_p, N, R, UEpositions, APpositions, mode):
    '''Create the initial footprints for the K-footprints algorithm
        INPUT>
        :param K: number of users
        :param L: number of APs
        :param tau_p: Number of orthogonal pilots
        :param N: number of antennas at the BS
        :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                                matrix between  AP l and UE k (normalized by noise variance)
        :param UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                            where the real part is the horizontal position and the imaginary part is the
                            vertical position
        :param APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                            where the real part is the horizontal position and the imaginary part is the
                            vertical position
        :param mode: select the footprint initialization mode
        OUTPUT>
        R_UEs: matrix of dimensions KxNxN that contains the normalized correlation matrices of users
        R_footprints: matrix of dimensions number_of_footprintsxNxN that contains the normalized correlation matrices of footprints
        number_footprints: number of initial footprints
        '''

    # to store the correlation matrices for the UEs in a different structure
    R_UEs = np.zeros((K, N, L*N), dtype=complex)

    for k in range(K):
        reshaped_R = np.array(R[:, :, :, k].reshape(N, L * N))
        R_UEs[k, :, :] = reshaped_R/linalg.norm(reshaped_R)

    match mode:
        case 'randomlocations':
            print('mode: randomlocations')

        case 'randomUEs':
            print('mode: randomUEs')

        case 'disimilarUEs':
            # print('mode: disimilarUEs')

            # number of initial footprints
            number_footprints = int(K / tau_p)
            # number_footprints = 3

            first_user = np.random.randint(0, K - 1)

            initial_users = [first_user]
            
            for _ in range(number_footprints-1):
                R_initial = np.array(sum([R_UEs[k, :, :] for k in initial_users]), dtype=complex)
                similarity_matrix = (R_UEs.reshape(K, -1) @ R_initial.reshape(-1, 1).conjugate()).real
                initial_users.append(np.argmin(similarity_matrix))

            # to store the correlation matrices for the initial footprints
            R_footprints = np.zeros((number_footprints, N, L*N), dtype=complex)
            
            for idx, user in enumerate(initial_users):
                R_footprints[idx, :, :] = R_UEs[user, :, :]

    return R_UEs, R_footprints, number_footprints
