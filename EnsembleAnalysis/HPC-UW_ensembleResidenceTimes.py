import numpy as np
import os
import pandas as pd
from scipy import stats

import fnmatch
import glob

import underworld as uw
from natsort import natsorted

from mpi4py import MPI

import pyvista as pv

import h5py

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
output = forwadModel_dir + r'ensembleResults/ResidenceTimes/'


if uw.mpi.rank == 0:
    if not os.path.exists(output):
        os.makedirs(output)

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

### load the points on first rank then scatter evenly across CPUs
if rank == 0:
    ### load the data
    x        = np.ascontiguousarray(np.load(forwadModel_dir + 'allStreamlinePoints.npy')[:,0])
    y        = np.ascontiguousarray(np.load(forwadModel_dir + 'allStreamlinePoints.npy')[:,1])
    z        = np.ascontiguousarray(np.load(forwadModel_dir + 'allStreamlinePoints.npy')[:,2])
    mat      = np.repeat(np.arange(0,14, dtype='float64'), (len(x)/14))

    ### store all the data
    allRTData     = np.zeros((x.shape[0], its))*np.nan
    allVelData    = np.zeros((x.shape[0], its))*np.nan
    allLengthData = np.zeros((x.shape[0], its))*np.nan


    split      = np.array_split(mat,size,axis = 0)
    split_size = [len(split[i]) for i in range(len(split))]
    split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]

else:
    mat = None
    x   = None
    y   = None
    z   = None

    split = None
    split_size = None
    split_disp = None

comm.barrier()

split_size = comm.bcast(split_size, root = 0)
split_disp = comm.bcast(split_disp, root = 0)

mat_local = np.zeros(split_size[rank])
x_local   = np.zeros(split_size[rank])
y_local   = np.zeros(split_size[rank])
z_local   = np.zeros(split_size[rank])

comm.barrier()

### scatter points and mat index across CPUs

comm.Scatterv([mat, split_size, split_disp, MPI.DOUBLE], mat_local, root=0)
comm.Scatterv([x, split_size, split_disp, MPI.DOUBLE], x_local, root=0)
comm.Scatterv([y, split_size, split_disp, MPI.DOUBLE], y_local, root=0)
comm.Scatterv([z, split_size, split_disp, MPI.DOUBLE], z_local, root=0)

points = np.zeros((mat_local.shape[0], 3))

points[:,0] = x_local
points[:,1] = y_local
points[:,2] = z_local

comm.barrier()



#### load mesh in pyvista

# define bounding box
xmin, xmax, ymin, ymax = -955637.8812, 1034362.2443650428, 6342298.2975, 8922298.39436168
zmin, zmax = -8000.0, 1200.0

# resolution
dx, dy, dz = 10e3, 10e3, 100
# dx, dy, dz = 10e3, 10e3, 50
Nx, Ny, Nz = int((xmax-xmin)/dx), int((ymax-ymin)/dy), int((zmax-zmin)/dz)



Xcoords = np.linspace(xmin,xmax,Nx)
Ycoords = np.linspace(ymin,ymax,Ny)

grid = pv.StructuredGrid()

grid.points = h5py.File(glob.glob(forwadModel_dir + 'mesh*')[0], 'r')['vertices'][:]

grid.dimensions = Nx+1, Ny+1, Nz+1

comm.barrier()


### a for number of reps for each it
a = 0
### j for column in np array to update
j = 0

### loop over the ensemble loading in the velocity field on each cpu and calculate streamlines
for file in natsorted(glob.glob(checkpoints + 'velocityField*')):
    grid["velocity"] = h5py.File(file, 'r')['data'][:]
    grid.set_active_vectors("velocity")


    length_local = np.zeros((len(points)))*np.nan
    velocity_local = np.zeros((len(points)))*np.nan
    residenceTime_local = np.zeros((len(points)))*np.nan



    for i in (range(0,len(points))):
        #### points to generate streamlines
        points0 = pv.StructuredGrid()
        points0.points = points[i]
        #### create streamlines
        streamlines = grid.streamlines_from_source(source=points0, vectors='velocity', min_step_length=0.01, max_step_length=0.5, terminal_speed = 0.000000000001, max_error=0.000001, initial_step_length=0.2)

        ### determine the length
        length_local[i] = (streamlines.length)
        ### determine average velocity along streamline
        velocity_local[i] = (np.linalg.norm((streamlines['velocity'])))

        residenceTime_local[i] = ((streamlines.length) / (np.mean(np.linalg.norm(streamlines['velocity'], axis=1))))

    ### gather data back on main CPU to save the streamline data
    ### Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(points), root=0))

    comm.barrier()

    if rank == 0:
    ### creates dummy data on all nodes to store the surface
        length         = np.zeros((sum(sendcounts)), dtype='float64')*np.nan
        velocity       = np.zeros((sum(sendcounts)), dtype='float64')*np.nan
        residenceTime  = np.zeros((sum(sendcounts)), dtype='float64')*np.nan
        matIndex       = np.zeros((sum(sendcounts)), dtype='float64')*np.nan
    else:
        length        = None
        velocity      = None
        residenceTime = None
        matIndex      = None


    comm.barrier()
    ### gather values back on the main CPU
    comm.Gatherv(sendbuf=length_local, recvbuf=(length, sendcounts), root=0)
    comm.Gatherv(sendbuf=velocity_local, recvbuf=(velocity, sendcounts), root=0)
    comm.Gatherv(sendbuf=residenceTime_local, recvbuf=(residenceTime, sendcounts), root=0)
    comm.Gatherv(sendbuf=mat_local, recvbuf=(matIndex, sendcounts), root=0)

    comm.barrier()

    if rank == 0:
        data      = np.zeros((matIndex.shape[0], 4))*np.nan
        data[:,0] = length
        data[:,1] = velocity
        data[:,2] = residenceTime
        data[:,3] = matIndex

        if (np.isnan(data).any()):
            sys.exit("NaN detected in resident time data")
        else:
            np.save(output + os.path.basename(file)[:-3] +'-residenceTimeData', data)

        start = j
        end   = j+reps[a]

        allRTData[:,start:end]  = np.transpose([residenceTime] * reps[a])
        allVelData[:,start:end] = np.transpose([velocity] * reps[a])
        allLengthData[:,start:end] = np.transpose([length] * reps[a])

        j+=reps[a]
        a+= 1

        print(file + ' imported')

    comm.barrier()


if rank == 0:
    if (np.isnan(allRTData).any()):
        sys.exit("NaN detected in all resident time data")
    np.save(output + 'allResidenceTimeData', allRTData)

    if (np.isnan(allVelData).any()):
        sys.exit("NaN detected in velocity data")
    np.save(output + 'allVelocityData', allVelData)

    if (np.isnan(allLengthData).any()):
        sys.exit("NaN detected in velocity data")
    np.save(output + 'allLengthData', allLengthData)
