import numpy as np
import numpy.linalg as alg
import math
import scipy.linalg as spalg
import matplotlib.pyplot as plt
from itertools import product

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

def correlationNormalized_grid(R_fixed, N, UE_positions):
    APposition = 500 + 500j

    grid = np.zeros((100, 100))

    for idxi, i in enumerate(range(0, 1000, 10)):
        for idxj, j in enumerate(range(0, 1000, 10)):
            UE_mobil = complex(i, j)
            UE_mobil_angle = np.angle(UE_mobil - APposition)

            R_mobil = localScatteringR(N, UE_mobil_angle)
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

def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))