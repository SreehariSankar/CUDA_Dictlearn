"""
CUDA IMPLEMENTATION

"""

import pickle
import numpy as np
import warnings
import load.loaddata as ld
import print.printfigure as pf
import matplotlib.pyplot as plt
from sporco.cupy import dictlrn
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
# Suppress warnings like future deprecations
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    """This is the main file that calls load, run and print"""
    "Load data"
    # loadData object, you can pass the name of the file to load as a string

    data1 = ld.LoadData('Data_Vib_Freq_parsed.mat')

    data2 = ld.LoadData('Data_Ind_Amp_parsed_2.mat')

    # Indices that we want to load
    indices1=np.arange(3,247,3)

    # indices1=[11,12,13,14,15]
    # 10hz increasing amplitude indices1=[33,43,53,63]

    # 50HZ INCREASING AMPLITUDE INDICES
    # indices1=[106,116,126,136,146]



    data_ra, data_sa, data_pc = data1.neuralresponse(indices1)
    stimulus_ra, stimulus_sa, stimulus_pc = data1.stimulus(indices1)
    print(data_ra.shape, "Data_ra shape")
    print(data_sa.shape, "Data_sa shape")
    print(data_pc.shape, "Data_pc shape")

    online_cdl=dictlrn.onlinecdl
    opt=online_cdl.OnlineConvBPDNDictLearn.Options({
                'Verbose': True, 'ZeroMean': False, 'eta_a': 10.0,
                'eta_b': 20.0, 'DataType': np.float32,
                'CBPDN': {'rho': 5.0, 'AutoRho': {'Enabled': True},
                    'RelaxParam': 1.8, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
                    'FastSolve': False, 'DataType': np.float32}})
    lmbda = 0.2

    if not cupy_enabled():
        print('CuPy/GPU device not available: running without GPU acceleration\n')
    else:
        id = select_device_by_load()
        info = gpu_info()
        if info:
            print('Running on GPU %d (%s)\n' % (id, info[id].name))

    data_ra_new=[]
    for i in range(1,int(data_ra.shape[0]/1000)):
        data_ra_new.append(data_ra[(i-1)*1000:i*1000,:])
    data_ra=np.asarray(data_ra_new)
    print(data_ra.shape,"New data_ra shape")

    stimulus_ra_new=[]
    for i in range(1,int(stimulus_ra.shape[0]/1000)):
        stimulus_ra_new.append(stimulus_ra[(i-1)*1000:i*1000,:])
    stimulus_ra=np.asarray(stimulus_ra_new)


    d = online_cdl.OnlineConvBPDNDictLearn(np2cp(np.load('Data_81sets/v_ra')), lmbda, opt,dimK=0,dimN=1)
    d.init_vars(data_ra[0],dimK=0)

    d.display_start()
    for it in range(len(data_ra)):
        d.solve(np2cp(data_ra[it]),dimK=0)
        d.solve(np2cp(stimulus_ra[it]),dimK=0)

    d.display_end()
    D1 = cp2np(d.getdict())
    print("OnlineConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

    pickle.dump(D1, open('OUTPUT_CUDA_RA', 'wb'))

    '''
    # ==========================================================================================================    
    # THIS SECTION IS TO BE USED TO CONCATENATE MULTIPLE INPUTS ACROSS MULTIPLE DATA FILES----------------------
    
    data_ra, data_sa, data_pc = data1.neuralresponse(indices1)
    x,y,z=data_ra.shape,data_sa.shape,data_pc.shape

    stimulus_ra, stimulus_sa, stimulus_pc = data1.stimulus(indices1)
    a, b, c = stimulus_ra.shape, stimulus_pc.shape, stimulus_sa.shape
    
    data_ra1, data_sa1, data_pc1 = data2.neuralresponse(indices2)
    x1,y1,z1=data_ra1.shape,data_sa1.shape,data_pc1.shape
       
    stimulus_ra1, stimulus_sa1, stimulus_pc1 = data2.stimulus(indices2)
    a1,b1,c1=stimulus_ra1.shape,stimulus_pc1.shape,stimulus_sa1.shape

    print(data_ra.shape,"Data_ra shape, 1ST SET")
    print(data_sa.shape,"Data_sa shape, 1ST SET")
    print(data_pc.shape,"Data_pc shape, 1ST SET")
    print(data_ra1.shape,"Data_ra shape, 2ND SET")
    print(data_sa1.shape,"Data_sa shape, 2ND SET")
    print(data_pc1.shape,"Data_pc shape, 2ND SET")
    
    data_ra=np.append(data_ra,data_ra1).reshape(x[0]+x1[0],x[1])
    data_sa=np.append(data_sa,data_sa1).reshape(y[0]+y1[0],y[1])
    data_pc=np.append(data_pc,data_pc1).reshape(z[0]+z1[0],z[1])
    stimulus_ra=np.append(stimulus_ra,stimulus_ra1).reshape(a[0]+a1[0],a[1])
    stimulus_sa=np.append(stimulus_sa,stimulus_sa1).reshape(c[0]+c1[0],c[1])
    stimulus_pc=np.append(stimulus_pc,stimulus_pc1).reshape(b[0]+b1[0],b[1])
    
    print(data_ra.shape,"RA:After append")
    print(data_sa.shape,"SA:After append")
    print(data_pc.shape,"PC:After append")
    
    # THIS SECTION IS TO BE USED TO CONCATENATE MULTIPLE INPUTS ACROSS MULTIPLE DATA FILES----------------------
    # ==========================================================================================================
    '''





    "Save the data"
    # pickle.dump(item.u_ra, open('data//u_ra', 'wb'))
    # pickle.dump(item.u_sa, open('data/u_sa', 'wb'))
    # pickle.dump(item.u_pc, open('data//u_pc', 'wb'))
    #
    # pickle.dump(item.v_ra, open('data//v_ra', 'wb'))
    # pickle.dump(item.v_sa, open('data/v_sa', 'wb'))
    # pickle.dump(item.v_pc, open('data//v_pc', 'wb'))
    #
    # pickle.dump(item.u_s_ra, open('data//u_s_ra', 'wb'))
    # pickle.dump(item.u_s_sa, open('data/u_s_sa', 'wb'))
    # pickle.dump(item.u_s_pc, open('data//u_s_pc', 'wb'))
    #
    # fig = pf.PrintFigure(indices1)
    #
    # plt.plot(stimulus_ra)
    # plt.show()
    #
    # datara=(np.load('data/u_ra',allow_pickle=True)@np.load('data/v_ra',allow_pickle=True))
    # datara[datara>0.433]=1
    # datara[datara<=0.433]=0
    # for i in range(1,int(stimulus_ra.shape[0]/1000)):
    #     plt.plot(stimulus_ra[(i-1)*1000:i*1000,:])
    #     plt.savefig(open("stim_"+str(i-1)+".png",'wb'),clear=True)
    #
    # print((np.sum(np.abs(datara-data_ra))/np.sum(data_ra)),"Error")
    # errors=[]
    # for i in range(1,int(datara.shape[0]/1000)):
    #     errors.append((np.sum(np.abs(datara[(i-1)*1000:i*1000,:] - data_ra[(i-1)*1000:i*1000,:])) / np.sum(datara[(i-1)*1000:i*1000,:])))
    # errors=np.asarray(errors)
    # idx=np.argsort(errors,kind='mergesort')
    # print(idx)