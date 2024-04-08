import numpy
import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random

def channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p):
    """Generate the channel realizations and estimations of these channels for all the UEs in the entire network.
    The channels are assumed to be correlated Rayleigh fading and the MMSE estimator is used.
    INPUT>
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel between
    UE k and AP l (normalized by noise variance)
    :param nbrOfRealizations: number of channel realizations
    :param L: number of Aps
    :param K: number of UEs
    :param N: number of antennas per AP
    :param tau_p: number of orthogonal pilots
    :param pilotIndex: vector containing the pilot assigned to each UE
    :param p: Uplink transmit power per UE (same for everyone)

    OUTPUT>
    Hhat: matrix with dimensions L*N x nbrOfRealizations x K where (:, n, k) is the estimated collective channel to
                    UE k in the channel realization n
    H: matrix with dimensions L*N x nbrOfRealizations x K where (:, n, k) is the true realization of the collective
                    channel to UE k in the channel realization n.
    B: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel estimate
                    between AP l and UE k (normalized by noise variance)
    C: matrix with dimension N x N x L x K where (:,:,l,k) is the spatial correlation matrix of the channel estimation
                    error between AP l and UE k (normalized by noise variance)
    """

    # set random seed
    np.random.seed(0)

    # Generate uncorrelated Rayleigh fading channel realizations
    H = np.random.randn(L*N, nbrOfRealizations, K) + 1j*np.random.randn(L*N, nbrOfRealizations, K)

# ######
#     from scipy.io import loadmat
#     mat_data = loadmat('mat.mat')
#     # Extract the 4D matrix
#     H = mat_data['H']
# #####

    # Go through all channels and apply the spatial correlation matrices
    for l in range(L):
        for k in range(K):
            # Apply correlation to the uncorrelated channel realizations
            Rsqrt = spalg.sqrtm(R[:, :, l, k])          # square root of the correlation matrix
            H[l*N:(l+1)*N, :, k] = np.sqrt(0.5)*Rsqrt@H[l*N:(l+1)*N, :, k]

    # Perform channel estimation
    # Store identity matrix of size NxN
    eyeN = np.identity(N)

    # Generate realizations of normalized noise
    Np = np.sqrt(0.5) * (np.random.randn(N, nbrOfRealizations, L, tau_p) + 1j * np.random.randn(N, nbrOfRealizations, L, tau_p))

# ######
#     from scipy.io import loadmat
#     mat_data = loadmat('Np.mat')
#     # Extract the 4D matrix
#     Np = mat_data['Np']
# #####



    # Prepare to store results
    Hhat = np.zeros((L*N, nbrOfRealizations, K), dtype=complex)
    B = np.zeros((R.shape), dtype=complex)
    C = np.zeros((R.shape), dtype=complex)

    # Go through all the APs
    for l in range(L):

        # Go through all the pilots
        for t in range(tau_p):

            # Compute processed pilot signal for all the UEs that use pilot t with an additional scaling factor
            # \sqrt(tau_p)
            yp = np.sqrt(p) * tau_p * np.sum(H[l*N: (l+1) * N, :, t == pilotIndex], axis=2) + np.sqrt(tau_p)*Np[:, :, l, t]

            # Compute the matrix that is inverted in the MMSE estimator
            PsiInv = (p * tau_p * np.sum(R[:, :, l, t == pilotIndex], axis=2) + eyeN)

            # Go through all the UEs that use pilot t
            pilotsharingUEs, = np.where(t == pilotIndex)
            if len(pilotsharingUEs)>0:
                for k in pilotsharingUEs:

                    # Compute the MSE estimate
                    RPsi = R[:, :, l, k]@alg.inv(PsiInv)
                    Hhat[l * N: (l+1)*N, :, k] = np.sqrt(p) * RPsi @ yp

                    # Compute the spatial correlation matrix of the estimation
                    B[:, :, l, k] = p * tau_p * RPsi @ R[:, :, l, k]

                    # Compute the spatial correlation matrix of the estimation error
                    C[:, :, l, k] = R[:, :, l, k] - B[:, :, l, k]

    return Hhat, H, B, C
