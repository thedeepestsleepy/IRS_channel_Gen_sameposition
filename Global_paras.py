## Parameters 
# Number of users
import numpy as np
import os
Num_User = 1
Num_Ant_BS = 64
Num_Ant_IRS = 100
Num_Data = 10000
batchS = 500

##OFDM paras
#Number of subcarries
K = 128
#length of the cyclic prefix (CP)
CP = (K*2) // 4
#Pilot length
P = K
#P = 10
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
if P < K:
    pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
    dataCarriers = np.delete(allCarriers, pilotCarriers)

else:  # K = P
    pilotCarriers = allCarriers
    dataCarriers = []

mu = 2
payloadBits_per_OFDM = K * mu

Data_Num = Num_Data
BS = batchS
Nt = Num_Ant_BS
Ni = Num_Ant_IRS
Nr = Num_User
NumofPilots = K
NumofSignal = K
NumofData = Num_Data

Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K * mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')
Pilots_matrix = np.tile(bits,(Data_Num,1))