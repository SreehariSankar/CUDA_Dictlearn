import numpy as np
import h2o4gpu.solvers.factorization as f
import timeit
import scipy
a=np.load('Data_test/data_ra_test',allow_pickle=True)
a=scipy.sparse.coo_matrix(a,dtype=np.float32)

start=timeit.timeit()
n=900
dict_learner=f.FactorizationH2O(n,0.4)
dict_learner.fit(a)
x=dict_learner.XT
y=dict_learner.thetaT
print(x.shape,y.shape)
stop=timeit.timeit()

print(stop-start)


start=timeit.timeit()


stop=timeit.timeit()
print(stop-start)