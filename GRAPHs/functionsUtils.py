import numpy as np
import numpy.linalg as alg
import math
import scipy.linalg as spalg
import matplotlib.pyplot as plt
from itertools import product
import plotly.express as px
import pickle

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


def drawingSetup(UEpositions, APpositions, colorIndex, title, squarelength):
    """
    INPUT>
    :param UEpositions: list of lists containing positions of UEs
    :param APpositions: list of lists containing positions of APs
    :param colorIndex: index of color for each UE
    :param title: title of the plot
    :param squarelength: side of the cell
    """

    # create a custom color palette for up to 10 orthogonal pilots
    custom_colors = np.array(['red', 'dodgerblue', 'green', 'orchid', 'aqua', 'orange', 'lime', 'black', 'pink', 'yellow']*10)

    # pilot assignment graph
    plt.scatter(UEpositions.real, UEpositions.imag, c=custom_colors[colorIndex], marker='*')
    plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^', s=8)
    plt.title(title)
    for i, txt in enumerate(range(len(UEpositions))):
        plt.annotate(txt, (UEpositions[i].real, UEpositions[i].imag))
    plt.xlim([0, squarelength])
    plt.ylim([0, squarelength])
    plt.show()

def drawing3Dvectors(UEvectorMatrix, colorIndex, title):
    # # create a custom color palette for up to 10 orthogonal pilots
    custom_colors = np.array(
        ['red', 'dodgerblue', 'green', 'orchid', 'aqua', 'orange', 'lime', 'black', 'pink', 'yellow'] * 10)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(UEvectorMatrix[:,0], UEvectorMatrix[:,1], UEvectorMatrix[:,2], c=custom_colors[colorIndex], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    for i, txt in enumerate(range(UEvectorMatrix.shape[0])):
        ax.text(UEvectorMatrix[i,0], UEvectorMatrix[i,1], UEvectorMatrix[i,2], str(i))

    plt.show()

def save_results(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_results(filename):
    with open(filename, 'rb') as inp:
        results = pickle.load(inp)
    return results
