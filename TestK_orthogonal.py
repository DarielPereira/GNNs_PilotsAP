from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
from functionsSetup import localScatteringR, db2pow
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.linalg import null_space

ASD_varphi = math.radians(10)
antennaSpacing = 0.5
distanceVertical = 10
noiseFigure = 7
B = 20*10**6
noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm
alpha = 36.7                # pathloss parameters for the path loss model
constantTerm = -30.5


tau_p = 2
N = 20
K = 5
L = 1
pilotIndex = np.array([0, 1, 1, 1, 0], dtype=int)

UEs = [200+300j, 400+420j, 500+400j, 600+600j, 600+400j]
APposition = 500+500j

R_full = np.zeros((N, N, L, K), dtype=complex)
R_normalized = np.zeros((N, N, L, K), dtype=complex)

UE_gainOverNoisedB = np.zeros(K)
D = np.ones((L, K))


for k in range(K):
    UE_angle = np.angle(UEs[k] - APposition)
    UE_distance = np.sqrt(distanceVertical**2+np.abs(APposition-UEs[k])**2)
    UE_gainOverNoisedB[k] = constantTerm - alpha * np.log10(UE_distance) - noiseVariancedBm
    R_k = np.array(localScatteringR(N, UE_angle, ASD_varphi, antennaSpacing))
    R_full[:, :, 0, k] = db2pow(UE_gainOverNoisedB[k])*R_k
    R_normalized[:, :, 0, k] = R_k/linalg.norm(R_k)

R_k_vectorized = R_normalized[:, :, 0, 0].reshape(1, -1)
ns_k = null_space(R_k_vectorized.conjugate())
projMatrix_ns_k = ns_k@linalg.inv(ns_k.conjugate().T@ns_k)@ns_k.conjugate().T

distances = np.zeros(K)
distances2 = np.zeros(K)

for k in range(K):
    proj_ns_k = projMatrix_ns_k@R_normalized[:, :, 0, k].reshape(-1, 1)
    distances[k] = linalg.norm(R_normalized[:, :, 0, k].reshape(-1, 1) - proj_ns_k)
    distances2[k] = linalg.norm(proj_ns_k)

system_NMSE, UEs_NMSE, average_NMSE = functionComputeNMSE_uplink(D, tau_p, N, K, L, R_full, pilotIndex)

print("{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(*UEs_NMSE.real))
