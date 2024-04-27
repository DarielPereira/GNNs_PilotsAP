from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
from functionsSetup import localScatteringR, db2pow
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as linalg

ASD_varphi = math.radians(5)
antennaSpacing = 0.5
distanceVertical = 10
noiseFigure = 7
B = 20*10**6
noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm
alpha = 36.7                # pathloss parameters for the path loss model
constantTerm = -30.5

p = 100
tau_p = 10
N = 200
K = 2
L = 1
pilotIndex = np.array([0, 0], dtype=int)

UE_fixed = 200+300j
APposition = 500+500j
R = np.zeros((N, N, L, K), dtype=complex)
D = np.ones((L, K))

UE_fixed_angle = np.angle(UE_fixed - APposition)
UE_fixed_distance = np.sqrt(distanceVertical**2+np.abs(APposition-UE_fixed)**2)
UE_fixed_gainOverNoisedB = constantTerm - alpha * np.log10(UE_fixed_distance) - noiseVariancedBm
UE_fixed_R = localScatteringR(N, UE_fixed_angle, ASD_varphi, antennaSpacing)
R[:, :, 0, 0] = db2pow(UE_fixed_gainOverNoisedB)*UE_fixed_R

eigenvalues, eigenvectors = np.linalg.eig(np.array(UE_fixed_R,dtype=complex))
index = np.argsort(eigenvalues)
eigenvectors = eigenvectors[:, index]
eigenvalues = eigenvalues[index]
UE_fixed_eigenVector = eigenvectors.flatten()

NMSe_fixed_values = np.zeros((100, 100))
product_eigenVectors = np.zeros((100, 100))
product_Rs_norm = np.zeros((100, 100))
product_Rs_norm_norm = np.zeros((100, 100))
product_Rs = np.zeros((100, 100))
NMSE_uncorrelated = np.zeros((100, 100))
NMSE_combination_test = np.zeros((100, 100))

for idxi, i in enumerate(range(0, 1000, 10)):
    for idxj, j in enumerate(range(0, 1000, 10)):
        UE_mobil = complex(i, j)
        UE_mobil_angle = np.angle(UE_mobil - APposition)
        UE_mobil_distance = np.sqrt(distanceVertical ** 2 + np.abs(APposition - UE_mobil) ** 2)
        UE_mobil_gainOverNoisedB = constantTerm - alpha * np.log10(UE_mobil_distance) - noiseVariancedBm

        UE_mobil_R = localScatteringR(N, UE_mobil_angle, ASD_varphi, antennaSpacing)
        R[:, :, 0, 1] = db2pow(UE_mobil_gainOverNoisedB) * UE_mobil_R
        system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)
        NMSe_fixed_values[idxj, idxi] = UEs_NMSE[0]

        eigenvalues, eigenvectors = np.linalg.eig(np.array(UE_mobil_R, dtype=complex))
        index = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, index]
        eigenvalues = eigenvalues[index]
        UE_mobil_eigenVector = eigenvectors.flatten()

        # product_eigenVectors[idxj, idxi] = np.abs(np.vdot(UE_fixed_eigenVector, UE_mobil_eigenVector))
        product_Rs[idxj, idxi] = np.abs(np.vdot(np.array(R[:, :, 0, 0]), np.array(R[:, :, 0, 1])))
        product_Rs_norm[idxj, idxi] = np.abs(np.vdot(np.array(UE_fixed_R), np.array(UE_mobil_R)))
        product_Rs_norm_norm[idxj, idxi] = (np.abs(np.vdot(np.array(UE_fixed_R), np.array(UE_mobil_R)))/
                                       (linalg.norm(np.array(UE_fixed_R))*linalg.norm(np.array(UE_mobil_R))))

        NMSE_uncorrelated[idxj, idxi] =  (1 - db2pow(UE_fixed_gainOverNoisedB) * tau_p * p /
                                           (1+db2pow(UE_fixed_gainOverNoisedB)*p*tau_p+db2pow(UE_mobil_gainOverNoisedB)*tau_p*p))

        NMSE_combination_test[idxj, idxi] = ((1 - db2pow(UE_fixed_gainOverNoisedB) * tau_p * p /
                                           (1+db2pow(UE_fixed_gainOverNoisedB)*p*tau_p+db2pow(UE_mobil_gainOverNoisedB)*tau_p*p))
                                          * product_Rs_norm_norm[idxj, idxi])


np.savez(f'./Grid',
                 grid_NMSEs=NMSe_fixed_values, grid_productCorrelations=product_Rs_norm_norm, UE_position=UE_fixed)



x = np.arange(0, 1000, 10)
y = np.arange(0, 1000, 10)

fig, ax0 = plt.subplots()
im0 = plt.pcolormesh(x, y, NMSe_fixed_values[:-1, :-1])
ax0.set_title('NMSE correlated')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
fig.colorbar(im0, ax=ax0)
plt.show()

# fig1, ax1 = plt.subplots()
# im1 = plt.pcolormesh(x, y, product_eigenVectors[:-1, :-1])
# ax1.set_title('Scalar product between eigen_vectors')
# plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
# fig.colorbar(im1, ax=ax1)
# plt.show()

fig2, ax2 = plt.subplots()
im2 = plt.pcolormesh(x, y, product_Rs_norm[:-1, :-1])
ax2.set_title('Scalar product between normalized correlation matrices')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
fig.colorbar(im2, ax=ax2)
plt.show()

fig3, ax3 = plt.subplots()
im3 = plt.pcolormesh(x, y, NMSE_uncorrelated[:-1, :-1])
ax3.set_title('NMSE uncorrelated case')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
fig.colorbar(im3, ax=ax3)
plt.show()

fig4, ax4 = plt.subplots()
im4 = plt.pcolormesh(x, y, product_Rs[:-1, :-1])
ax4.set_title('Scalar product between correlation matrices')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
fig.colorbar(im4, ax=ax4)
plt.show()

fig5, ax5 = plt.subplots()
im5 = plt.pcolormesh(x, y, NMSE_combination_test[:-1, :-1])
ax5.set_title('combination of angular info in Rs and uncorrelated NMSE')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
fig.colorbar(im5, ax=ax5)
plt.show()

fig6, ax6 = plt.subplots()
im6 = plt.pcolormesh(x, y, product_Rs_norm_norm[:-1, :-1])
ax6.set_title('Scalar product between truly normalized correlation matrices')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
fig.colorbar(im6, ax=ax6)
plt.show()




