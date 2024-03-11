from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
from functionsSetup import localScatteringR, db2pow, drawPilotAssignment
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.linalg import null_space


def correlationNormalized_grid(R_fixed, N, UE_positions):
    APposition = 500 + 500j

    grid = np.zeros((100, 100))

    for idxi, i in enumerate(range(0, 1000, 10)):
        for idxj, j in enumerate(range(0, 1000, 10)):
            UE_mobil = complex(i, j)
            UE_mobil_angle = np.angle(UE_mobil - APposition)

            R_mobil = localScatteringR(N, UE_mobil_angle, ASD_varphi, antennaSpacing)
            R_mobil = R_mobil / np.linalg.norm(R_mobil)

            grid[idxj, idxi] = (np.abs(np.vdot(np.array(R_fixed), np.array(R_mobil))) )

    x = np.arange(0, 1000, 10)
    y = np.arange(0, 1000, 10)

    fig, ax0 = plt.subplots()
    im0 = plt.pcolormesh(x, y, grid[:-1, :-1])
    ax0.set_title('R product')
    plt.scatter(UE_positions.real, UE_positions.imag, marker='+', color='r')
    for i, txt in enumerate(range(len(UE_positions))):
        plt.annotate(txt, (UE_positions[i].real, UE_positions[i].imag))
    fig.colorbar(im0, ax=ax0)
    plt.show()

ASD_varphi = math.radians(5)
antennaSpacing = 0.5
distanceVertical = 10
noiseFigure = 7
B = 20*10**6
noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm
alpha = 36.7                # pathloss parameters for the path loss model
constantTerm = -30.5

squarelength = 1000

# UE_positions = np.array([200+300j, 400 + 420j, 500+400j, 600+600j, 600+405j])
np.random.seed(0)
UE_positions = (np.random.rand(100) + 1j*np.random.rand(100))*squarelength
APposition = 500 + 500j


tau_p = 2
N = 200
K = 100
L = 1
pilotIndex = np.array([0, 1, 1, 1, 0], dtype=int)

number_beams = 10

initial_angles_deg = np.linspace(-85, 85, number_beams+1)[:-1]
initial_angles = np.radians(initial_angles_deg)

R_beams = np.zeros((number_beams, N, N), dtype=complex)

R_UEs = np.zeros((K, N, N), dtype=complex)

for k in range(K):
    for l in range(L):  # Go through all APs
        angle_k = np.angle(UE_positions[k] - APposition)
        R_k = localScatteringR(N, angle_k, ASD_varphi, antennaSpacing)
        R_UEs[k, :, :] = R_k/np.linalg.norm(R_k)

beam_allocation = [ [] for _ in range(number_beams) ]


for beam in range(number_beams):
    R = localScatteringR(N, initial_angles[beam], ASD_varphi, antennaSpacing)
    R_beams[beam, :, :] = R / np.linalg.norm(R)

# R_combined = np.sum(R_beams, axis=0)
# correlationNormalized_grid(R_combined, N, UE_positions)

correlation_factors = np.zeros((K, number_beams))
for k in range(len(UE_positions)):
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
pilotIndex = np.zeros((K), int)
for beam in range(number_beams):
    for user in beam_allocation[beam]:
        pilotIndex[user] = beam

drawPilotAssignment(UE_positions, APposition, pilotIndex, title='UE Clustering ')
#

updating = True
while updating:
    # beam_allocation = new_beam_allocation
    new_beam_allocation = [[] for _ in range(number_beams)]
    correlation_factors = np.zeros((K, number_beams))
    for k in range(len(UE_positions)):
        # correlation_factors = np.zeros(number_beams)
        for beam in range(number_beams):
            correlation_factors[k, beam] = np.abs(np.vdot(np.array(new_R_beams[beam, :, :]), np.array(R_UEs[k, :, :])))
        matching_beam = np.argmax(correlation_factors[k, :])
        new_beam_allocation[matching_beam].append(k)
    new_R_beams = []
    for beam in range(number_beams):
        if new_beam_allocation[beam]:
            new_R_beams.append(sum([R_UEs[allocated_user, :, :] for allocated_user in new_beam_allocation[beam]]))
    number_beams = len(new_R_beams)
    new_R_beams = np.array(new_R_beams)

    new_beam_allocation = list(filter(None, new_beam_allocation))

    # to draw
    pilotIndex = np.zeros((K), int)
    for beam in range(number_beams):
        for user in new_beam_allocation[beam]:
            pilotIndex[user] = beam

    drawPilotAssignment(UE_positions, APposition, pilotIndex, title='UE Clustering ')
    #

    if new_beam_allocation == beam_allocation:
        updating = False
    else:
        beam_allocation = new_beam_allocation


    # for beam in range(len(new_R_beams)):
    #     correlationNormalized_grid(new_R_beams[beam], N, UE_positions)

