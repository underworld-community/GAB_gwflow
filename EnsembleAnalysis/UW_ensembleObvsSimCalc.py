import numpy as np
import os
import pandas as pd
from scipy import stats
import glob
from natsort import natsorted
import underworld as uw
import rasterio
import rioxarray
import pyvista as pv
import h5py
import open3d as o3d
import sys


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### define path files are stored in

### where the required files are stored

data_dir = '/home/565/bk2562/models/GAB_data/data/'

forwadModel_dir = r'/scratch/q97/bk2562/GAB_MHmodel_test/GAB_forwardModel/'


### where velocity field is stored
checkpoints = forwadModel_dir + r'checkpoints/'

### where to save the esemble stats
output = forwadModel_dir + r'ensembleResults/'

raster_dir = data_dir + r"/GAB_surfaces/GeoTiff/"



if uw.mpi.rank == 0:
    if not os.path.exists(output):
        os.makedirs(output)

comm.barrier()

reps = None
its  = None

df_layers = pd.read_csv(data_dir + "GAB_surfaces/UW-GAB_layer_parameters.csv")
LayerNames = df_layers['Name Aquifer/Aquitard']


if uw.mpi.rank == 0 :

    #### load in the datasts
    complete_set = pd.read_csv(forwadModel_dir + 'MH_output.csv', index_col=[0])[4:]


    ### count duplicates in list. Calculates how many times to repeat file if it appears multiple times in the ensemble

    res=complete_set.iloc[:,0:14].reset_index().groupby(complete_set.iloc[:,0:14].columns.tolist())["index"].agg(list).reset_index().rename(columns={"index": "duplicated"})
    res.index=res["duplicated"].str[0].tolist()
    res["duplicated"]=res["duplicated"].str[1:]
    res['no. of dups'] = complete_set.iloc[:,0:14].groupby(complete_set.iloc[:,0:14].columns.tolist()).size().reset_index().rename(columns={0:'count'})['count'].values

    res = res.sort_index()

    reps = res['no. of dups'].values

    its = len(complete_set)

    print('loaded in data')

    print('its = {}, sum(reps) = {}'.format(its, sum(reps)))

reps = uw.mpi.comm.bcast(reps, root=0)
its  = uw.mpi.comm.bcast(its, root=0)

comm.barrier()


if its != sum(reps):
    sys.exit("Sum of reps does not equal number of iterations")






comm.barrier()

#### CPU pressure data
localArraySize                = np.array_split(np.load(natsorted(glob.glob(checkpoints + 'pressureMisfit*'))[0]),size,axis = 0)[rank].shape[0]
localPressureData             = np.zeros((localArraySize, its))*np.nan





comm.barrier()

# %%
### a for number of reps for each it
a = 0
### j for column in np array to update
j = 0

for file in natsorted(glob.glob(checkpoints + 'pressureMisfit*')):
    ### load in pressure observation dataset
    data         = np.array_split(np.load(file),size,axis = 0)[rank]

    start = j
    end   = j+reps[a]
    #### add into dataframe
    localPressureData[:,start:end] = np.transpose([data] * reps[a])

    if uw.mpi.rank ==0:
        print(file + ' imported')

    j+=reps[a]
    a+= 1

if (np.isnan(localPressureData).any()):
    sys.exit("NaN detected in calculation")


localMeanPressure = localPressureData.mean(axis=1)
localMinPressure  = localPressureData.min(axis=1)
localMaxPressure  = localPressureData.max(axis=1)
localPressurestd  = localPressureData.std(axis=1)


sendcounts0 = np.array(comm.gather(localMeanPressure.shape[0], root=0))
sendcounts1 = np.array(comm.gather(localMinPressure.shape[0], root=0))
sendcounts2 = np.array(comm.gather(localMaxPressure.shape[0], root=0))
sendcounts3 = np.array(comm.gather(localPressurestd.shape[0], root=0))


meanPressure = None
minPressure  = None
maxPressure  = None
stdPressure  = None

if rank == 0:
    ### creates dummy data on all nodes to store the flux data
    meanPressure = np.zeros((sum(sendcounts0)), dtype='float64')*np.nan
    minPressure  = np.zeros((sum(sendcounts1)), dtype='float64')*np.nan
    maxPressure  = np.zeros((sum(sendcounts2)), dtype='float64')*np.nan
    stdPressure  = np.zeros((sum(sendcounts3)), dtype='float64')*np.nan


comm.Gatherv(sendbuf=localMeanPressure, recvbuf=(meanPressure, sendcounts0), root=0)
comm.Gatherv(sendbuf=localMinPressure, recvbuf=(minPressure, sendcounts1), root=0)
comm.Gatherv(sendbuf=localMaxPressure, recvbuf=(maxPressure, sendcounts2), root=0)
comm.Gatherv(sendbuf=localPressurestd, recvbuf=(stdPressure, sendcounts3), root=0)

if rank == 0:
    if (np.isnan(meanPressure).any()):
        sys.exit("NaN detected in sim mean pressure dataset")
    else:
        np.save(output+ 'simMeanPressure', meanPressure)

    if (np.isnan(minPressure).any()):
        sys.exit("NaN detected in sim min pressure dataset")
    else:
        np.save(output+ 'simMinPressure', minPressure)

    if (np.isnan(maxPressure).any()):
        sys.exit("NaN detected in sim max pressure dataset")
    else:
        np.save(output+ 'simMaxPressure', maxPressure)
    if (np.isnan(maxPressure).any()):
        sys.exit("NaN detected in sim max pressure dataset")
    else:
        np.save(output+ 'simPressureSTD', stdPressure)
