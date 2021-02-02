import numpy as np
from matplotlib import pyplot as plt
## Parameters 
#Load Global papas
# Generate time domain channel H_G and H_r

from Global_paras import *

# Distance between BS and the center of user areas /m
# the center of the user areas coordinate
Lroom = 200 
Wroom= 30
User_position = np.zeros((Num_User,2))
pt = np.array([0.0,0.0])
k1 = np.array([1.0,0])
k2 = np.array([0,1.0])
# the range of the user areas /m
R = 10
##
# Generate the location of users coordinate

for index in range(Num_User):
    pt = np.array([0.0,0.0])
    r = np.random.random_sample()
    theta = np.random.random_sample()*2*np.math.pi
    px = r * np.cos(theta)
    py = r * np.sin(theta)
    pt = np.array([Lroom,Wroom]) + px*k1 + py*k2
    User_position[index,:] = pt

np.save('./Data/Location/User_position.npy',User_position)


###End of the coordinate generation
# print(Pt)
# plt.plot(Pt[:,0],Pt[:,1],'ro')
# plt.xlim([Lroom-R,Lroom+R])
# plt.ylim([Wroom-R,Wroom+R])
# plt.show()

## generate path loss

# calculate distance
#BS and IRS location
BS_positon = np.array([0.0,0.0])
IRS_positon = np.array([200.0,0.0])
#the distance between the BS and the IRS
BS_IRS_Dis = np.sqrt(np.sum(np.power(\
    np.abs(BS_positon-IRS_positon),2)))
# the loss of the BS-to-IRS path alpha_l
BS_IRS_Loss = 35.6 +22*np.log10(BS_IRS_Dis)


IRS_User_Loss = np.zeros((Num_User,1))
for index in range(Num_User):
    temp = User_position[index,:] - IRS_positon
    dis_UI = np.sqrt(np.sum(np.power( np.abs(temp),2)))
    IRS_User_Loss[index,:] = 35.6 +22*np.log10(dis_UI)
# the loss of the IRS-to-User path \beta_l
##
IRS_User_Loss = IRS_User_Loss+BS_IRS_Loss #  sum loss of BS-IRS-USer link


# direct link loss
BS_User_Loss = np.zeros((Num_User,1))
for index in range(Num_User):
    temp = User_position[index,:]-BS_positon
    dis_BU = np.sqrt(np.sum(np.power( np.abs(temp),2)))
    BS_User_Loss[index,:] = 35.6 +22*np.log10(dis_BU)

BS_User_Loss = BS_User_Loss+BS_IRS_Loss

IRS_User_Loss = IRS_User_Loss - BS_User_Loss

## generate the channel 
## Rician loss
Noise = -170 + 10*np.log10(180*1000)
Noise = float(Noise)
# path_ direct
#(Num_user,1)
BS_User_Noise = 10**((-(BS_User_Loss+Noise))/10)
#path_Irs
IRS_User_Noise = 10**((-Noise-IRS_User_Loss)/10)

pd = np.sqrt(BS_User_Noise)
#(Num_user,Num_Ant_BS)
pd = np.tile(pd,[1,Num_Ant_BS])

ps = np.sqrt(IRS_User_Noise)
ps = np.tile(ps,[1,Num_Ant_IRS])
# initial theta_init and channel Hd
Hd_w = np.zeros((Num_Data,Num_User,Num_Ant_BS))
theta_init = np.zeros((Num_Data,Num_Ant_IRS,1))
for index in range(Num_Data):
    Hd_temp = np.sqrt(1/2)* np.random.randn(Num_User,Num_Ant_BS)+\
        1j*np.random.randn(Num_User,Num_Ant_BS)
    Hd_w[index,:,:]= Hd_temp*pd
    theta_temp = np.exp(np.random.rand(Num_Ant_IRS,1)*2*np.math.pi)
    theta_init[index,:,:] = theta_temp
#print('log shape is:',theta_init.shape,Hd_w.shape)

eb = 10
eb2 = 1/(1+eb)
eb1 = np.sqrt(1-eb2)
eb2 = np.sqrt(eb2)
AoD_BS = 0.45
AoA_IRS = 0.45
AoD_IRS = np.random.random((Num_User,1))
G_sig = np.zeros((Num_Data,Num_Ant_IRS,Num_Ant_BS),np.complex)
Hr_sig = np.zeros((Num_Data,Num_User,Num_Ant_IRS),np.complex)
for index in range(Num_Data):
    G_sig[index,:,:] = np.sqrt(0.5)*(np.random.randn(\
        Num_Ant_IRS,Num_Ant_BS)+ 1j*np.random.randn(Num_Ant_IRS,Num_Ant_BS))
    Hr_sig[index,:,:] = np.sqrt(0.5)*(np.random.randn(\
        Num_User,Num_Ant_IRS)+ 1j*np.random.randn(Num_User,Num_Ant_IRS))
print('end')



### ULA
def channelresponse_ULA(\
    azimuth_angle,NumofAntanna):
    steering = np.zeros(int(NumofAntanna),dtype = complex)
    for index_m in range(int(np.sqrt(NumofAntanna))):
        steering[index_m] = np.exp(1j*np.pi*\
            (index_m*np.sin(azimuth_angle)))
    steering = steering / (np.sqrt(NumofAntanna))
    return steering
### UPA
def channelresponse_UPA(\
    azimuth_angle,elevater_angel,NumofAntanna):
    steering = np.zeros(int(NumofAntanna),dtype = complex)
    for index_m in range(int(np.sqrt(NumofAntanna))):
        for index_n in range(int(np.sqrt(NumofAntanna))):
            steering[index_m+index_n] = np.exp(1j*np.pi*\
                (index_m*np.sin(azimuth_angle)*\
                    np.sin(elevater_angel) + \
                        index_n*np.cos(elevater_angel)))
    steering = steering / (np.sqrt(NumofAntanna))
    return steering

At_G = channelresponse_ULA(AoD_BS,Num_Ant_BS) # shape is (Num_User,)
Ar_IRS = channelresponse_ULA(AoA_IRS,Num_Ant_IRS)
H_G = np.matmul(np.expand_dims(Ar_IRS,-1),np.expand_dims(At_G,0))
H_G = eb1*H_G + eb2*G_sig
At_IRS = np.zeros((Num_User,Num_Ant_IRS),dtype = complex)
for index in range(Num_User):
    At_IRS[index,:] = channelresponse_ULA(AoD_IRS[index],Num_Ant_IRS) 
H_r = eb1*At_IRS + eb2*Hr_sig # shape is (Num_Data,Num_User,Num_Ant_IRS)
print('end')

## up-link Phi initialization
Phi_up_init = channelresponse_UPA(AoD_BS,0,Num_Ant_IRS)


np.save('./Data/Channel/Channel_G.npy', H_G)
np.save('./Data/Channel/Channel_Hr.npy',H_r)
np.save('./Data/Channel/Channel_Hd.npy',Hd_w)
np.save('./Data/Channel/Phase_uplink_init.npy',Phi_up_init)
print('shape check',H_G.shape,H_r.shape,Hd_w.shape,Phi_up_init.shape)
