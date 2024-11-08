from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
from functionsSetup import localScatteringR, db2pow
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from functionsUtils import db2pow, localScatteringR, drawingSetup, drawing3Dvectors
from functionsSetup import generateSetup
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink

ASD_varphi = math.radians(5)
antennaSpacing = 0.5
distanceVertical = 10
noiseFigure = 7
B = 20*10**6
noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm
alpha = 36.7                # pathloss parameters for the path loss model
constantTerm = -30.5
squarelength = 1000
sigma_sf = 4

p = 100
tau_p = 10
N = 4
K = 2
L=100

pilotIndex = np.array([0, 0], dtype=int)
AP_spacing = 100

ASD_varphi = math.radians(10)
ASD_theta = math.radians(15)
nbrOfRealizations = 1

gainOverNoisedB, distances, R_, APpositions, UEpositions = generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta,
                                                               nbrOfRealizations, seed=1)
L = gainOverNoisedB.shape[0]

UE_fixed = UEpositions[0, 0]

Grid_dotProduct = np.zeros((100, 100))
Grid_NMSE = np.zeros((100, 100))
R = np.zeros((N, N, L, 2), dtype=complex)

for idxi, i in enumerate(range(0, 1000, 10)):
    for idxj, j in enumerate(range(0, 1000, 10)):

        UEpositions[1, 0] = complex(i, j)

        distances[:, 1] = np.sqrt(distanceVertical ** 2 + np.abs(APpositions - UEpositions[1, 0]) ** 2)[:, 0]
        gainOverNoisedB[:, 1] = constantTerm - alpha * np.log10(distances[:, 1]) - noiseVariancedBm + (np.sqrt(sigma_sf**2))*np.random.randn(L)

        for l in range(L):  # Go through all APs
            angletoUE_varphi = np.angle(UEpositions[1, 0] - APpositions[l])

            # Generate the approximate spatial correlation matrix using the local scattering model by scaling
            # the normalized matrices with the channel gain
            R_[:, :, l, 1] = db2pow(gainOverNoisedB[l, 1]) * localScatteringR(N, angletoUE_varphi, ASD_varphi,
                                                                             antennaSpacing)

        R[:, :, :, 0] = R_[:, :, :, 0]
        R[:, :, :, 1] = R_[:, :, :, 1]

        footprint_fixed = R[:, :, :, 0].reshape(N, N*L)
        footprint_mobile = R[:, :, :, 1].reshape(N, N*L)

        # gainMatrix = db2pow(gainOverNoisedB.T)
        # # norm_gainMatrix = gainMatrix / (gainMatrix.max(axis=1).reshape(-1, 1))
        # # norm_gainMatrix = gainMatrix / (gainMatrix.max(axis=0).reshape(1, -1))
        # norm_gainMatrix = gainMatrix / linalg.norm(gainMatrix, axis=1).reshape(-1, 1)

         # norm_gainMatrix = gainMatrix

        # dot_product = np.abs(np.vdot(np.array(norm_gainMatrix[0]), np.array(norm_gainMatrix[1])))
        dot_product = (np.abs(np.vdot(np.array(footprint_fixed), np.array(footprint_mobile)))/
                       (linalg.norm(np.array(footprint_fixed))*linalg.norm(np.array(footprint_mobile))))

        Grid_dotProduct[idxj, idxi] = dot_product


        D = np.ones((L, 2))
        # D = np.zeros((L,2))
        # # every user is served at least by its master AP
        # for k in range(2):
        #     # Determine the master AP for UE k by looking for the AP with best channel condition
        #     master = np.argmax(gainOverNoisedB[:, k])
        #
        #     # serve user k by its master
        #     D[master, k] = 1

        # system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot = functionComputeNMSE_uplink(D, tau_p,
        #                                                                                        N, 2, L, R, pilotIndex)
        # Grid_NMSE[idxj, idxi] = UEs_NMSE[0]

x = np.arange(0, 1000, 10)
y = np.arange(0, 1000, 10)

fig, ax0 = plt.subplots()
im0 = plt.pcolormesh(x, y, Grid_dotProduct[:-1, :-1])
ax0.set_title('dot product')
plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^', s=8)
fig.colorbar(im0, ax=ax0)
plt.show()

np.savez(f'./GRAPHs/VARIABLES_SAVED/Grid',
                 grid_product=Grid_dotProduct, UE_position=UE_fixed, AP_positions = APpositions)

#
# fig4, ax4 = plt.subplots()
# im4 = plt.pcolormesh(x, y, Grid_NMSE[:-1, :-1])
# ax4.set_title('NMSE correlated')
# plt.scatter(UE_fixed.real, UE_fixed.imag, marker='+', color='r')
# plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^', s=8)
# fig.colorbar(im4, ax=ax4)
# plt.show()

#
# fig1, ax1 = plt.subplots()
# plt.plot(norm_gainMatrix[0].T, '*')
# plt.show()
#
# fig2 = plt.figure(figsize =(14, 9))
# ax2 = plt.axes(projection ='3d')
# ax2.scatter(APpositions.real, APpositions.imag, gainMatrix[0, :].reshape(-1, 1))
# plt.show()
#
# x = np.outer(np.linspace(0, 990, 100), np.ones(100))
# y = x.copy().T
# # Creating color map
# my_cmap = plt.get_cmap('hot')
#
# fig3 = plt.figure(figsize =(14, 9))
# ax3 = plt.axes(projection ='3d')
# ax3.plot_surface(x, y, Grid_dotProduct, cmap=my_cmap)
# plt.show()

print('end')


