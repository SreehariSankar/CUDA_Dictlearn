import numpy as np
import cupy as cp
import timeit
import sporco.admm.cmod as cmod
import sporco.cupy.admm as admm
import sporco.admm
import pickle
import warnings
import matplotlib.pyplot as plt
import load.loaddata as ld
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)


data1 = ld.LoadData('Data_Vib_Freq_parsed.mat')
# Indices that we want to load
indices1=np.arange(3,247,3)
data_ra, data_sa, data_pc = data1.neuralresponse(indices1)
stimulus_ra, stimulus_sa, stimulus_pc = data1.stimulus(indices1)
print(data_ra.shape, "Data_ra shape")
print(data_sa.shape, "Data_sa shape")
print(data_pc.shape, "Data_pc shape")




data_ra_new=[]
for i in range(1,int(data_ra.shape[0]/1000)):
    data_ra_new.append(data_ra[(i-1)*1000:i*1000,:])
data_ra=np.asarray(data_ra_new)
print(data_ra.shape,"New data_ra shape")

data_sa_new=[]
for i in range(1,int(data_sa.shape[0]/1000)):
    data_sa_new.append(data_sa[(i-1)*1000:i*1000,:])
data_sa=np.asarray(data_sa_new)
print(data_sa.shape,"New data_sa shape")

data_pc_new=[]
for i in range(1,int(data_pc.shape[0]/1000)):
    data_pc_new.append(data_pc[(i-1)*1000:i*1000,:])
data_pc=np.asarray(data_pc_new)
print(data_pc.shape,"New data_ra shape")

stimulus_ra_new=[]
for i in range(1,int(stimulus_ra.shape[0]/1000)):
    stimulus_ra_new.append(stimulus_ra[(i-1)*1000:i*1000,:])
stimulus_ra=np.asarray(stimulus_ra_new)

stimulus_sa_new=[]
for i in range(1,int(stimulus_sa.shape[0]/1000)):
    stimulus_sa_new.append(stimulus_sa[(i-1)*1000:i*1000,:])
stimulus_sa=np.asarray(stimulus_sa_new)

stimulus_pc_new=[]
for i in range(1,int(stimulus_pc.shape[0]/1000)):
    stimulus_pc_new.append(stimulus_pc[(i-1)*1000:i*1000,:])
stimulus_pc=np.asarray(stimulus_pc_new)


data_ra=np2cp(data_ra)
data_sa=np2cp(data_sa)
data_pc=np2cp(data_pc)
stimulus_ra=np2cp(stimulus_ra)
stimulus_sa=np2cp(stimulus_sa)
stimulus_pc=np2cp(stimulus_pc)

# v_ra=np.load('Data_81sets/v_ra',allow_pickle=True)
# v_sa=np.load('Data_81sets/v_sa',allow_pickle=True)
# v_pc=np.load('Data_81sets/v_pc',allow_pickle=True)
v_ra=cp.eye(data_ra[1].shape[1])
v_sa=cp.eye(data_ra[1].shape[1])
v_pc=cp.eye(data_ra[1].shape[1])


A0=cp.zeros(v_ra.shape)
B0=cp.zeros(v_ra.shape)
for lll in range(10):
    u = []
    time_start = timeit.timeit()

    for i in range(len(data_ra)):
        s=data_ra[i]
        y=stimulus_ra[i]
        estimator=admm.bpdn.BPDN(v_ra.T,s.T,0.6)
        u_ra=estimator.solve()
        # print(u_ra.shape,"U shape")
        x=np2cp(u_ra)
        u.append(x)

    time_stop = timeit.timeit()
    print(time_stop - time_start,"Done creating Us")

    time_start = timeit.timeit()
    for p in u:
        A0+=p@p.T
        B0+=u_ra@s
        # FEATURES X NEURONS
    for j in range(v_ra.shape[1]):
        if A0[j][j]==0:
            print(A0[j][j]==0,"!")
            exit()
        vv =(1/A0[j][j])*(B0[:,j]-(v_ra.T@A0[:,j])+v_ra[:,j]*A0[j][j])
        v_ra[:,j]=vv/cp.linalg.norm(vv)
    time_stop = timeit.timeit()
    print(time_stop - time_start, lll,"-th iteration done")

    print(cp.linalg.norm(v_ra.T@u[0]-data_ra[0].T))
pickle.dump(cp2np(v_ra),open('V_RA','wb'))
k=v_ra.T@u[0]
k[k>0.4]=1
k[k<=0.4]=0
plt.plot(k)
plt.show()