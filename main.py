import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as alg

##Setting Parameters
nbrOfSetups = 2                 #number of Monte-Carlo setups
nbrOfRealizations = 3           #number of channel realizations per setup

L = 100                         #number of APs
N = 4                           #number of antennas per AP

K = 40                          #number of UEs

tau_c = 200                     #length of coherence block
tau_p = 10                      #lenghth of pilot sequences
prelogFactor = 1-tau_p/tau_c    #uplink data transmission prelog factor
