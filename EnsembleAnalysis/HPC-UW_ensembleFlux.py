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

### Required functions

def create_surfaces(i):
    ''' create the points that represents the base of the layer '''
    ### only run once at beginning
    ### extract the required surfaces
    top = rasterio.open(raster_dir + df_layers['Layer name'].iloc[i] + '.tiff').read(1, masked=True)
    bottom = rasterio.open(raster_dir + df_layers['Layer name'].iloc[i+1] + '.tiff').read(1, masked=True)

    tiffFile = rioxarray.open_rasterio(raster_dir + df_layers['Layer name'].iloc[i] + '.tiff')

    thickness = np.abs(top.data - bottom.data)
    thickness[thickness == 0] = np.nan
    base_coords = (top - thickness)

    xx, yy = np.meshgrid(tiffFile.x.values, tiffFile.y.values)

    ### extracts the base surface
    points0 = np.array((xx.flatten(), yy.flatten(), base_coords.data.flatten())).T



    points = np.array((xx.flatten(), yy.flatten(), base_coords.data.flatten())).T[~np.isnan(thickness.flatten())]#[::20]

    mask = np.all(np.isin(points0, points), axis=1)


    point_data = points0[mask]

    surface = pv.PolyData(point_data)

    ### caculates the normals
    pointSet = o3d.geometry.PointCloud()
    pointSet.points = o3d.utility.Vector3dVector(points)
    pointSet.estimate_normals()

    surface['normals'] = np.asarray(pointSet.normals)

    return surface

def probe_grid(grid, surface, velocityField):
    ''' probe the points that has been created to represent the base of the layer '''

    data = grid.probe(surface)


    data['surfaceVelocity'] = np.array([np.dot(data['normals'][v], data[velocityField][v]) for v in range(len(data[velocityField]))])



    return data['surfaceVelocity']




def calcNormals(points):
    pointSet = o3d.geometry.PointCloud()
    pointSet.points = o3d.utility.Vector3dVector(points)
    pointSet.estimate_normals()
    return np.asarray(pointSet.normals)



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


### create each of the surfaces on each CPU
### only run once to create the surfaces
num_of_surfaces = np.arange(1,14,1)

surfaces = None
if uw.mpi.rank == 0:
    surfaces = [create_surfaces(v) for v in num_of_surfaces]





comm.barrier()

### create list of velocity field and then split across CPUs
velocityFieldFiles        = natsorted(glob.glob(checkpoints + 'velocityField*'))

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

### loop over the ensemble loading in the velocity field on each cpu and calculate the flux
velocityFieldsLocal = list(np.array_split(velocityFieldFiles,size,axis = 0)[rank])
repsLocal           = np.array_split(reps,size,axis = 0)[rank]



if rank == 0:
    allLayerData = np.zeros((its, len(num_of_surfaces)))*np.nan
else:
    allLayerData = None

for i in num_of_surfaces:

    surface = None
    if uw.mpi.rank == 0:
        surface = surfaces[i-1]

    surface = uw.mpi.comm.bcast(surface, root=0)

    flux = []

    for file, rep in zip(velocityFieldsLocal, repsLocal):
        grid["velocity"] = h5py.File(file, 'r')['data'][:]
        grid.set_active_vectors("velocity")

        flux.extend(np.repeat(probe_grid(grid, surface, 'velocity').sum(), rep))


    flux = np.squeeze(np.array(flux))



    sendcounts = np.array(comm.gather(len(flux), root=0))

    TotalFlux = None

    if rank == 0:
        ### creates dummy data on all nodes to store the flux data
        TotalFlux = np.zeros((sum(sendcounts)), dtype='float64')*np.nan

        if sum(sendcounts) != sum(reps):
            sys.exit("sendcounts != length of ensemble")


    comm.Gatherv(sendbuf=flux, recvbuf=(TotalFlux, sendcounts), root=0)


    if rank == 0:
        if (np.isnan(TotalFlux).any()):
            sys.exit("NaN detected in all flux data")
        else:
            np.save(output+ LayerNames[i]+'-FluxData', TotalFlux)

            allLayerData[:,i-1] = np.array(TotalFlux)

            print('finished layer ' + LayerNames[i])


if rank == 0:
    if (np.isnan(allLayerData).any()):
        sys.exit("NaN detected in all flux data")
    else:
        np.save(output+'allFluxData', allLayerData)
