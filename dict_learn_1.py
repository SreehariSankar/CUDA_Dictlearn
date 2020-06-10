import sporco.dictlearn.dictlearn as dictlearn
import sporco.cupy.admm.bpdn as bpdn
import sporco.admm.cmod as admm
import cupy as cp
import numpy as np

data_ra=np.load('Data_test/data_ra_test',allow_pickle=True)
v_ra=np.load('Data_81sets/v_ra',allow_pickle=True)

# REMEBER TO TRANSPOSE THE MATRICES IN THE FORMULATION
# v_ra.T @ u_ra.T = data_ra.T

x=bpdn.BPDN(np2cp(v_ra.T),np2cp(data_ra.T),0.4)
u_ra=x.
d=admm.CnstrMOD()