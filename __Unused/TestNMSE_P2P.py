from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
from functionsSetup import localScatteringR, db2pow
import numpy as np
import math
import matplotlib.pyplot as plt

ASD_varphi = math.radians(10)
antennaSpacing = 0.5
distanceVertical = 10
noiseFigure = 7
B = 20*10**6
noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm
alpha = 36.7                # pathloss parameters for the path loss model
constantTerm = -30.5


tau_p = 10
N = 20
K = 5
L = 1
pilotIndex = np.array([0, 1, 1, 1, 1], dtype=int)

UEs = [100+100j, 400 + 420j, 500+400j, 600+600j, 600+400j]
APposition = 500+500j

R = np.zeros((N, N, L, K), dtype=complex)
UE_gainOverNoisedB = np.zeros(K)
D = np.ones((L, K))

for k in range(K):
    UE_angle = np.angle(UEs[k] - APposition)
    UE_distance = np.sqrt(distanceVertical**2+np.abs(APposition-UEs[k])**2)
    UE_gainOverNoisedB[k] = constantTerm - alpha * np.log10(UE_distance) - noiseVariancedBm
    R[:, :, 0, k] = db2pow(UE_gainOverNoisedB[k])*localScatteringR(N, UE_angle, ASD_varphi, antennaSpacing)


system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

print("{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(*UEs_NMSE.real))
