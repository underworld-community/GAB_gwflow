import numpy as np
import os
import pandas as pd
from scipy import stats

import fnmatch
import glob

import underworld as uw

from mpi4py import MPI

import pyvista as pv

import h5py

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### define path files are stored in

### where the required files are stored
path = r'/scratch/q97/bk2562/ensembleData/'


### where velocity field is stored
checkpoints = r'/scratch/q97/bk2562/checkpoints/'
# checkpoints = r'GAB-Notebooks/HPC_runs/'

output = r'/scratch/q97/bk2562/ensembleResidenceTimes/'

if uw.mpi.rank == 0:
    if not os.path.exists(output):
        os.makedirs(output)

reps = None
its  = None

if uw.mpi.rank == 0 :
    df_layers = pd.read_csv(path + 'UW-GAB_layer_parameters.csv')

    df_layers['conductivity (m/day)'] = (df_layers['ogia conductivity min (m/day)'] + df_layers['ogia conductivity max (m/day)']) /2.

    df_layers['Hydrostratigraphy'] = df_layers.iloc[:,6]


    layerAttrs = df_layers[['mat index', 'Name Aquifer/Aquitard', 'conductivity (m/day)', "Hydrostratigraphy"]].iloc[1:-1].set_index('Name Aquifer/Aquitard').T.to_dict()


    #### load in the datasts

    test0 = pd.read_csv(path + 'MH_output0.csv')

    test1 = pd.read_csv(path +'MH_output1.csv')

    frames = [test0, test1]

    complete_set = pd.concat(frames)


    ### rename headings of df
    LayerNames = dict(zip(complete_set.iloc[:,1:15].columns,df_layers['Name Aquifer/Aquitard'].iloc[1:15]))

    complete_set = complete_set.rename(columns=LayerNames)

    complete_set = complete_set.reset_index()

    ### discard first 100 values due to low x_scale value
    complete_set = complete_set.iloc[100:]

    complete_set = complete_set.iloc[:,1:]


    ### count duplicates in list

    res=complete_set.iloc[:,1:15].reset_index().groupby(complete_set.iloc[:,1:15].columns.tolist())["index"].agg(list).reset_index().rename(columns={"index": "duplicated"})
    res.index=res["duplicated"].str[0].tolist()
    res["duplicated"]=res["duplicated"].str[1:]
    res['no. of dups'] = complete_set.iloc[:,1:15].groupby(complete_set.iloc[:,1:15].columns.tolist()).size().reset_index().rename(columns={0:'count'})['count'].values

    res = res.sort_index()

    reps = res['no. of dups'].values

    its = len(complete_set)

    print('loaded in data')

reps = uw.mpi.comm.bcast(reps, root=0)
its  = uw.mpi.comm.bcast(its, root=0)

### load the points on first rank then scatter evenly across CPUs
if rank == 0:
    ### a for number of reps for each it
    a = 0
    ### j for column in np array to update
    j = 0

    ### load the data
    x        = np.ascontiguousarray(np.load(checkpoints + 'allStreamlinePoints.npy')[:,0])
    y        = np.ascontiguousarray(np.load(checkpoints + 'allStreamlinePoints.npy')[:,1])
    z        = np.ascontiguousarray(np.load(checkpoints + 'allStreamlinePoints.npy')[:,2])
    mat      = np.repeat(np.arange(0,14, dtype='float64'), (len(x)/14))

    ### store all the data
    allData = np.zeros((len(x), its))


    split      = np.array_split(mat,size,axis = 0)
    split_size = [len(split[i]) for i in range(len(split))]
    split_disp = np.insert(np.cumsum(split_size), 0, 0)[0:-1]

else:
#Create variables on other cores
    a = None
    j = None
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

grid.points = h5py.File(glob.glob(checkpoints + 'mesh*')[0], 'r')['vertices'][:]

grid.dimensions = Nx+1, Ny+1, Nz+1

comm.barrier()


### loop over the ensemble loading in the velocity field on each cpu and calculate streamlines
for file in sorted(glob.glob(checkpoints + '*velocity*')):
    grid["velocity"] = h5py.File(file, 'r')['data'][:]
    grid.set_active_vectors("velocity")


    length_local = np.zeros((len(points)))
    velocity_local = np.zeros((len(points)))
    residenceTime_local = np.zeros((len(points)))



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
        length         = np.zeros((sum(sendcounts)), dtype='float64')
        velocity       = np.zeros((sum(sendcounts)), dtype='float64')
        residenceTime  = np.zeros((sum(sendcounts)), dtype='float64')
        matIndex       = np.zeros((sum(sendcounts)), dtype='float64')
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
        data      = np.zeros((matIndex.shape[0], 4))
        data[:,0] = length
        data[:,1] = velocity
        data[:,2] = residenceTime
        data[:,3] = matIndex

        np.save(output + os.path.basename(file)[:-3] +'-residenceTimeData', data)



        for x in range(0, reps[a]):
            allData[:,j] = data[:,2]
            j +=1

        a+= 1

    comm.barrier()


if rank == 0:
    np.save(output + 'allResidenceTimeData', allData)
