#!/usr/bin/env python
# coding: utf-8
# %%

# # GAB model: groundwater + heat flow
#
# Each layer corresponds to the region between two surfaces as defined in `UW-GAB_layer_parameters.csv`. Each layer is assigned a "material index" which is used to map the thermal and hydraulic properties:
#
# - $k_h$: hydraulic conductivity (m/s)
# - $\phi$: porosity
# - $k_T$: thermal conductivity (W/m/K)
# - $H$: rate of heat production (W/m$^3$)

# %%


import underworld as uw

import numpy as np
import pandas as pd
import os
import csv


from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree
from scipy import optimize

import underworld.visualisation as vis
import matplotlib.pyplot as plt


import argparse
from time import time

from mpi4py import MPI
comm = MPI.COMM_WORLD

### additional requirements, may need to be installed
import xarray as xr
import rioxarray



startTiming   = time()

# %%


### convert between units
from pint import UnitRegistry
u = UnitRegistry()


# %%


hydraulicConductivityOnly = True

HPC_run = True

### set BURNIN to 0 if not using MH model to save all outputs from model
BURNIN = 100
NSIM   = 500
sigma  = 0.5



# %%


verbose = True
n_checkpoints = 1

###### Gauss point count, produces a total particles per cell of GPC ** model dimensions
#### Gauss point count of up to 4 seems to prevent the model from crashing.
if HPC_run == True:
    ### used for swarm for initial material distribution
    GPC = 4
    ### used for swarm that does the solve
    GPC_solver = 2
else:
    ### used for swarm for initial material distribution
    GPC = 2
    ### used for swarm that does the solve
    GPC_solver = 2




# ## Import datasets

# %%


### directory of stored data
if HPC_run == True:
    data_dir = '/home/565/bk2562/models/GAB_data/data/'
else:
    data_dir = '/home/jovyan/workspace/GAB-Notebooks/data/'

numpy_directory = data_dir + "GAB_surfaces/NumPy/"

geotiff_directory = data_dir + "GAB_surfaces/GeoTiff/"
# png_directory = "../data/GAB_surfaces/png/"


surface_filename_npz = numpy_directory + "{:s}.npz"
surface_filename_tiff = geotiff_directory + "{:s}.tiff"


df_layers = pd.read_csv(data_dir + "GAB_surfaces/UW-GAB_layer_parameters.csv")


# ## Set up model dimensions and parameters

# %%


Tmin = 298.0
Tmax = 500.0
Nx, Ny, Nz = 20,20,50 # global size

# define bounding box
xmin, xmax, ymin, ymax = -955637.8812, 1034362.2443650428, 6342298.2975, 8922298.39436168
zmin, zmax = -8000.0, 1200.0

# resolution
if HPC_run == True:
    dx, dy, dz = 10e3, 10e3, 100
else:
    dx, dy, dz = 40e3, 40e3, 1e3
# dx, dy, dz = 10e3, 10e3, 50
Nx, Ny, Nz = int((xmax-xmin)/dx), int((ymax-ymin)/dy), int((zmax-zmin)/dz)

if uw.mpi.rank == 0:
    print(f'Particles per cell: {GPC ** 3}')

    print("global number of elements in x,y,z {} | total number of elements = {}".format((Nx,Ny,Nz), Nx*Ny*Nz))

    print(f'Total particles: {Nx*Ny*Nz*(GPC**3)}')


### file directory to store data

if HPC_run == True:
    simulation_directory = "kh0_newkh0_{}dx-{}dy-{}dz_".format(dx/1e3, dy/1e3, dz/1e3) + "{}PPC/".format(GPC_solver**3)
else:
    simulation_directory = "../MH_timingTest/"

if uw.mpi.rank == 0:
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)


if uw.mpi.rank == 0 and n_checkpoints:
    if not os.path.exists(simulation_directory+'checkpoints/'):
        os.makedirs(simulation_directory+'checkpoints/')



# ## Create numpy arrays of datasets

# %%


matIndex = np.int32(df_layers['mat index'].iloc[1:-1].values)

### take average in log 10 space
kh0_log = (np.log10(df_layers['ogia conductivity min (m/day)'].iloc[1:-1].values) + np.log10(df_layers['ogia conductivity max (m/day)'].iloc[1:-1].values)) / 2.

### convert from log space and then convert from m/day to m/second
kh0 = (10**kh0_log * u.meter/u.day).to(u.meter/u.second).magnitude


kt0 = df_layers['thermal conductivity'].iloc[1:-1].values

dkt = 10*kt0

a = df_layers['a (T)'].iloc[1:-1].values


H0 = df_layers['Heat production (W/m3)'].iloc[1:-1].values

dH = 10*H0

porosityData = (df_layers['Porosity (%Vol)'].iloc[1:-1].values) / 100

LayerNames = list(df_layers['Layer name'].iloc[1:].values)

# dH = df_layers['Heat production error'][1:-1]


# ## Set up the mesh
#
# Initialise a Q1 finite element mesh and mesh variables.

# %%


deformedmesh = True
elementType = "Q1"
mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType),
                                 elementRes  = (Nx,Ny,Nz),
                                 minCoord    = (xmin,ymin,zmin),
                                 maxCoord    = (xmax,ymax,zmax))

gwHydraulicHead            = mesh.add_variable( nodeDofCount=1 )
temperatureField           = mesh.add_variable( nodeDofCount=1 )
temperatureField0          = mesh.add_variable( nodeDofCount=1 )
velocityField              = mesh.add_variable( nodeDofCount=3 )
heatProductionField        = mesh.add_variable( nodeDofCount=1 )

materialMesh               = mesh.add_variable( nodeDofCount=1 )

### used on the velocity field to caculate darcy flow
porosity                = mesh.add_variable( nodeDofCount=1 )


coords = mesh.data

Xcoords = np.unique(coords[:,0])
Ycoords = np.unique(coords[:,1])
Zcoords = np.unique(coords[:,2])
nx, ny, nz = Xcoords.size, Ycoords.size, Zcoords.size


# ## Deform mesh to surface & basement surfaces
#
# We want to deform the $z$-axis spacing so that the surface of the mesh is draped over the topography at the top and the basement rocks at the base.

# %%


topo_interp = None
basement_interp = None
if uw.mpi.rank == 0:


    with np.load(surface_filename_npz.format("AUSBATH09_AMG55_GDA94_500m_model_extent")) as npz:
        topo_interp = RegularGridInterpolator((npz['y'], npz['x']), np.flipud(npz['data']))

#     with np.load(surface_filename_npz.format("W910_BASEMENT_v2")) as npz:
#         basement_interp = RegularGridInterpolator((npz['y'], npz['x']), np.flipud(npz['data']))

    with rioxarray.open_rasterio(surface_filename_tiff.format("W910_BASEMENT_v2")) as npz:
        basement_interp = RegularGridInterpolator((np.flipud(npz.sel(band=1).y.data), npz.sel(band=1).x.data), np.flipud(npz.sel(band=1).data), bounds_error=False)


uw.mpi.comm.barrier()

topo_interp = uw.mpi.comm.bcast(topo_interp, root=0)
basement_interp = uw.mpi.comm.bcast(basement_interp, root=0)

uw.mpi.comm.barrier()


local_topography = topo_interp((mesh.data[:,1], mesh.data[:,0]))
local_basement = basement_interp((mesh.data[:,1], mesh.data[:,0]))

local_basement[np.isnan(local_basement)] = 0.

# ensure basement is at least as deep as topography!
local_basement = np.minimum(local_basement, local_topography)



# subtract a thickness buffer
dz_min = 2e3
local_basement -= dz_min

with mesh.deform_mesh():
    zcube = coords[:,2].reshape(nz,ny,nx)
    zcube_norm = zcube.copy()
    zcube_norm -= zmin
    zcube_norm /= zmax - zmin
    # difference to add to existing z coordinates
    dzcube1 = zcube_norm * -(zmax - local_topography.reshape(zcube.shape))
    dzcube0 = (1.0 - zcube_norm) * -(zmin - local_basement.reshape(zcube.shape))

    mesh.data[:,2] += dzcube1.ravel()
    mesh.data[:,2] += dzcube0.ravel()
    coords = mesh.data


# ## Set up the types of boundary conditions
#
# Set the left, right and bottom walls such that flow cannot pass through them, only parallel.
# In other words for groundwater head $h$:
#
# $ \frac{\partial h}{\partial x}=0$ : left and right walls
#
# $ \frac{\partial h}{\partial y}=0$ : bottom wall
#
# This is only solvable if there is topography or a non-uniform upper hydraulic head boundary condition.

# %%


topWall = mesh.specialSets["MaxK_VertexSet"]
bottomWall = mesh.specialSets["MinK_VertexSet"]

closeWall = mesh.specialSets["MinJ_VertexSet"]
farWall    = mesh.specialSets["MaxJ_VertexSet"]

leftWall   = mesh.specialSets["MinI_VertexSet"]
rightWall  = mesh.specialSets["MaxI_VertexSet"]

sideWalls = closeWall + farWall + leftWall + rightWall

gwPressureBC = uw.conditions.DirichletCondition( variable      = gwHydraulicHead,
                                               indexSetsPerDof = ( topWall   ) )

temperatureBC = uw.conditions.DirichletCondition( variable        = temperatureField,
                                                  indexSetsPerDof = (topWall+bottomWall))


# %%


gwHydraulicHead.data[:] = 0.

# # create a linear gradient [0, 1] top to bottom of mesh
znorm = mesh.data[:,2].copy()
znorm -= zmin
znorm /= (zmax-zmin)
# znorm = (zmin - local_topography)/zmin
linear_gradient = 1.0 - znorm

zmax - local_topography

# pressure and temperature initial conditions
initial_pressure = linear_gradient*(zmax-zmin)
initial_temperature = linear_gradient*(Tmax - Tmin) + Tmin
initial_temperature = np.clip(initial_temperature, Tmin, Tmax)


# %%


gwHydraulicHead.data[:]  = initial_pressure.reshape(-1,1)
temperatureField.data[:] = initial_temperature.reshape(-1,1)

# assign BCs (account for pressure of water below sea level)
sealevel = 0.0
seafloor = topWall[mesh.data[topWall,2] < sealevel]

gwHydraulicHead.data[topWall] = 0.
gwHydraulicHead.data[seafloor] = -((mesh.data[seafloor,2]-sealevel)*1.0).reshape(-1,1)
temperatureField.data[topWall] = Tmin
temperatureField.data[bottomWall] = Tmax


# ### Import water table surface

# %%


rgi_wt = None

if uw.mpi.rank == 0:
    with np.load(data_dir+ "GAB_surfaces/"+"water_table_surface.npz") as npz:
        wt = npz['data']
        wt_x = npz['x']
        wt_y = npz['y']

    rgi_wt = RegularGridInterpolator((wt_y, wt_x), wt)

uw.mpi.comm.barrier()

rgi_wt = uw.mpi.comm.bcast(rgi_wt, root=0)

uw.mpi.comm.barrier()



wt_interp = rgi_wt(mesh.data[topWall,0:2][:,::-1])
# gwHydraulicHead.data[topWall] = (-wt_interp * 1000.0 * 9.81).reshape(-1,1)

zCoordFn = uw.function.input()[2]
yCoordFn = uw.function.input()[1]
xCoordFn = uw.function.input()[0]

gwHydraulicHead.data[:] = zCoordFn.evaluate(mesh)
gwHydraulicHead.data[topWall] += (-wt_interp).reshape(-1,1)


# ## Set up particle swarm
#


# %%
# High PPC swarm to capture layers better within the model

swarm0         = uw.swarm.Swarm( mesh=mesh )
swarmLayout0   = uw.swarm.layouts.PerCellGaussLayout(swarm=swarm0,gaussPointCount=GPC)
swarm0.populate_using_layout( layout=swarmLayout0 )


# %%


materialIndex0           = swarm0.add_variable( dataType="int",    count=1 )
hydraulicDiffusivity0    = swarm0.add_variable( dataType="double", count=1 )
thermalDiffusivity0      = swarm0.add_variable( dataType="double", count=1 )
heatProduction0          = swarm0.add_variable( dataType="double", count=1 )
a_exponent0              = swarm0.add_variable( dataType="double", count=1 )



# # find cell centroids (parallel-safe, but need to accelerate this bit.)

# for cell in range(0, mesh.elementsLocal):
#     mask_cell = swarm.owningCell.data == cell
#     idx_cell  = np.nonzero(mask_cell)[0]
#     cell_centroid = swarm.data[idx_cell].mean(axis=0)
#     cellCentroid.data[idx_cell] = cell_centroid

# ### list comprehension version, may be some speed-up?
# cellCentroid.data[:] = np.array([(swarm.data[np.nonzero(swarm.owningCell.data == cell)[0]]).mean(axis=0) for cell in range(0, mesh.elementsLocal)])

# %%


# Create a swarm for the solver
# Each cell contains particles that _must_ be assigned isotropic thermal and hydraulic properties.
#
# > __Four__ particles per cell seems to prevent the model from crashing.

swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout   = uw.swarm.layouts.PerCellGaussLayout(swarm=swarm, gaussPointCount=GPC_solver)
swarm.populate_using_layout( layout=swarmLayout )


swarmVelocity  = swarm.add_variable( dataType="double", count=3 )

materialIndex           = swarm.add_variable( dataType="int",    count=1 )
fn_hydraulicDiffusivity = swarm.add_variable( dataType="double", count=1 )

thermalDiffusivity      = swarm.add_variable( dataType="double", count=1 )
heatProduction          = swarm.add_variable( dataType="double", count=1 )
a_exponent              = swarm.add_variable( dataType="double", count=1 )



# ## Import geological surfaces
#
# Assign a material index for all cell centroids which lie between two surfaces
#
# ## Assign material properties
#
# Use level sets to assign hydraulic diffusivities to a region on the mesh corresponding to any given material index.
#
# - $H$       : rate of heat production
# - $\rho$     : density
# - $k_h$     : hydraulic conductivity
# - $k_t$     : thermal conductivity
# - $\kappa_h$ : hydraulic diffusivity
# - $\kappa_t$ : thermal diffusivity

# %%


#### set all material to basement material initially
#index = df_layers['mat index'][df_layers['Base Surface'].str.contains('basement', case=False)].iloc[0]

### set all material to basement initially
index = matIndex[0]

### assign material variables
materialIndex0.data[:] = index

materialMesh.data[:] = index

### updates material on boudary to basement as there was a slight issue
materialMesh.data[sideWalls] = matIndex[-1]

materialIndex.data[:] = index

### add in layer properties
hydraulicDiffusivity0.data[:] = kh0[index] #((((df_layers['ogia conductivity max (m/day)'].iloc[index].values,df_layers['ogia conductivity min (m/day)'].iloc[index].values)/2.)  * u.meter/u.day).to(u.meter/u.second).magnitude)


thermalDiffusivity0.data[:] = kt0[index]#df_layers['thermal conductivity'].iloc[index]

a_exponent0.data[:] = a[index]#df_layers['a (T)'].iloc[index]

heatProduction0.data[:] = H0[index] #(df_layers['Heat production (W/m3)'].iloc[index])


porosity.data[:] = porosityData[index] #(df_layers['Porosity (%Vol)'].iloc[index]) / 100 # from % to a value between 0 and 1






# # Cell centroid method

# %%


# # In[19]:

# #

# mask_layer = np.ones(swarm.data.shape[0], dtype=bool)




# ## starting from the surface and going deeper with each layer
# ## skip topography as mesh is already deformed to topo
# for index in df_layers[1:-1].index[:]:
#     row = df_layers.loc[index]

#     # load surface
#     layer_interp = None
#     if uw.mpi.rank == 0:

#         with rioxarray.open_rasterio(surface_filename_tiff.format(row['Layer name'])) as npz:
#             layer_interp = RegularGridInterpolator((np.flipud(npz.sel(band=1).y.data), npz.sel(band=1).x.data), np.flipud(npz.sel(band=1).data), bounds_error=False)


#     uw.mpi.comm.barrier()

#     layer_interp = uw.mpi.comm.bcast(layer_interp, root=0)

#     uw.mpi.comm.barrier()

# #     ### interpolate surface to cell centroids
#     z_interp = layer_interp((cellCentroid.data[:,1], cellCentroid.data[:,0]))
#     z_interp_mesh = layer_interp((mesh.data[:,1], mesh.data[:,0]))

# ###     assign index to swarm particles which are below the current surface
# ###     if they are above the surface then we are done.


#     mask_layer = cellCentroid.data[:,2] < z_interp
#     mask_layer_mesh = mesh.data[:,2] < z_interp_mesh

#     ### assign material variables
#     materialIndex.data[mask_layer] = index


#     ### add in layer properties
#     ### Hydraulic conductivity in log10 scale
#     # fn_hydraulicDiffusivity.data[mask_layer] = ((np.mean((df_layers['ogia conductivity max (m/day)'].iloc[index],df_layers['ogia conductivity min (m/day)'].iloc[index]))  * u.meter/u.day).to(u.meter/u.second).magnitude)

#     hydraulicDiffusivity.data[mask_layer] = ((np.mean((df_layers['ogia conductivity max (m/day)'].iloc[index],df_layers['ogia conductivity min (m/day)'].iloc[index]))  * u.meter/u.day).to(u.meter/u.second).magnitude)

#     # print(hydraulicDiffusivity.data[mask_layer].min())


#     thermalDiffusivity.data[mask_layer] = df_layers['thermal conductivity'].iloc[index]

#     a_exponent.data[mask_layer] = df_layers['a (T)'].iloc[index]

#     ### heat production convert to micro
#     heatProduction.data[mask_layer] = (df_layers['Heat production (W/m3)'].iloc[index])

#     porosity.data[mask_layer_mesh] = (df_layers['Porosity (%Vol)'].iloc[index]) / 100 # from % to a value between 0 and 1


#     if uw.mpi.rank == 0:
#         print("Layer {:2d}  {}  ({})".format(index, row['Name Aquifer/Aquitard'], row['Layer name']))

# fn_hydraulicDiffusivity.data[:] = hydraulicDiffusivity.data[:].copy()

# if uw.mpi.rank ==0 and verbose:
#     print(np.isin(np.unique(fn_hydraulicDiffusivity.data), np.unique(hydraulicDiffusivity.data)))


# # Calculate mean of mixed cells
# The mean is calculated for cells with mixed hydraulic diffusivity values

# %%


mask_layer = np.ones(swarm.data.shape[0], dtype=bool)




## starting from the surface and going deeper with each layer
## skip topography as mesh is already deformed to topo
for index in matIndex:
    row = df_layers.loc[df_layers['mat index'] == index]

    # load surface
    layer_interp = None
    if uw.mpi.rank == 0:

        with rioxarray.open_rasterio(surface_filename_tiff.format(row['Layer name'].iloc[0])) as npz:
            layer_interp = RegularGridInterpolator((np.flipud(npz.sel(band=1).y.data), npz.sel(band=1).x.data), np.flipud(npz.sel(band=1).data), bounds_error=False)


    uw.mpi.comm.barrier()

    layer_interp = uw.mpi.comm.bcast(layer_interp, root=0)

    uw.mpi.comm.barrier()

#     ### interpolate surface to high res swarm0
    z_interp0 = layer_interp((swarm0.data[:,1], swarm0.data[:,0]))
    ### interpolate surface to high low res swarm
    z_interp = layer_interp((swarm.data[:,1], swarm.data[:,0]))
    ### interpolate surface to mesh
    z_interp_mesh = layer_interp((mesh.data[:,1], mesh.data[:,0]))

###     assign index to swarm particles which are below the current surface
###     if they are above the surface then we are done.


    mask_layer0     = swarm0.data[:,2] < z_interp0

    mask_layer      = swarm.data[:,2] < z_interp

    mask_layer_mesh = mesh.data[:,2] < z_interp_mesh

    ### assign material variables
    materialIndex0.data[mask_layer0] = index

    materialIndex.data[mask_layer] = index

    materialMesh.data[mask_layer_mesh] = index


    # ### add in layer properties
    # hydraulicDiffusivity0.data[mask_layer0] = ((((df_layers['ogia conductivity max (m/day)'].iloc[index],df_layers['ogia conductivity min (m/day)'].iloc[index])/2.)  * u.meter/u.day).to(u.meter/u.second).magnitude)


    # thermalDiffusivity0.data[mask_layer0] = df_layers['thermal conductivity'].iloc[index]

    # a_exponent0.data[mask_layer0] = df_layers['a (T)'].iloc[index]

    # heatProduction0.data[mask_layer0] = (df_layers['Heat production (W/m3)'].iloc[index])

    # porosity.data[mask_layer_mesh] = (df_layers['Porosity (%Vol)'].iloc[index]) / 100 # from % to a value between 0 and 1

    ### add in layer properties
    hydraulicDiffusivity0.data[mask_layer0] = kh0[index] #((((df_layers['ogia conductivity max (m/day)'].iloc[index].values,df_layers['ogia conductivity min (m/day)'].iloc[index].values)/2.)  * u.meter/u.day).to(u.meter/u.second).magnitude)


    thermalDiffusivity0.data[mask_layer0] = kt0[index]#df_layers['thermal conductivity'].iloc[index]

    a_exponent0.data[mask_layer0] = a[index]#df_layers['a (T)'].iloc[index]

    heatProduction0.data[index] = H0[index] #(df_layers['Heat production (W/m3)'].iloc[index])


    porosity.data[mask_layer_mesh] = porosityData[index] #(df_layers['Porosity (%Vol)'].iloc[index]) / 100 # from % to a value between 0 and 1


    if uw.mpi.rank == 0:
        print("Layer {:2d}  {}  ({})".format(index, row['Name Aquifer/Aquitard'].iloc[0], row['Layer name'].iloc[0]))

# determine mean of cells to produce uniform cells
fn_hydraulicDiffusivity.data[:,0] = np.repeat((np.mean((np.split(hydraulicDiffusivity0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)

thermalDiffusivity.data[:,0] = np.repeat((np.mean((np.split(thermalDiffusivity0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)
heatProduction.data[:,0] = np.repeat((np.mean((np.split(heatProduction0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)

a_exponent.data[:,0] = np.repeat((np.mean((np.split(a_exponent0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)




#
#

# %%


# +
swarm_topography = topo_interp((swarm.data[:,1],swarm.data[:,0]))
mesh_topography  = local_topography

beta = 9.3e-3
depth = -1.0*(materialIndex.swarm.data[:,2] - swarm_topography)
depth = np.clip(depth, 0.0, zmax-zmin)

depth_mesh = -1.0*(mesh.data[:,2] - mesh_topography)
depth_mesh = np.clip(depth_mesh, 0.0, zmax-zmin)

# +
Storage = 1.
rho_water = 1000.
c_water = 4e3
coeff = rho_water*c_water

if deformedmesh:
    g = uw.function.misc.constant((0.,0.,-1.))
else:
    g = uw.function.misc.constant((0.,0.,0.))

# g = uw.function.misc.constant((0.,0.,0.))

gwPressureGrad = gwHydraulicHead.fn_gradient

gMapFn = -g*rho_water*Storage


# %%


fn_thermalDiffusivity = thermalDiffusivity*(298.0/temperatureField)**a_exponent
fn_source = uw.function.math.dot(-1.0*coeff*velocityField, temperatureField.fn_gradient) + heatProductionField


# %%


# groundwater solver
gwadvDiff = uw.systems.SteadyStateDarcyFlow(
                                            velocityField    = velocityField, \
                                            pressureField    = gwHydraulicHead, \
                                            fn_diffusivity   = fn_hydraulicDiffusivity, \
                                            conditions       = [gwPressureBC], \
                                            fn_bodyforce     = (0.0, 0.0, 0.0), \
                                            voronoi_swarm    = swarm, \
                                            swarmVarVelocity = swarmVelocity)


# heatflow solver
heateqn = uw.systems.SteadyStateHeat( temperatureField = temperatureField,                                       fn_diffusivity   = fn_thermalDiffusivity,                                       fn_heating       = heatProduction,                                       conditions       = temperatureBC                                       )


# %%



### solve the groudwater flow
gwsolver = uw.systems.Solver(gwadvDiff)


### solve the heat flow
heatsolver = uw.systems.Solver(heateqn)


# %%


# find model elevation
topWall_xyz = uw.mpi.comm.allgather(mesh.data[topWall])
topWall_xyz = np.vstack(topWall_xyz)

# create downsampled interpolator for surface topography
interp_downsampled = LinearNDInterpolator(topWall_xyz[:,:2], topWall_xyz[:,2])

# topWall_swarm = topo_interp((topWall_xyz[:,1], topWall_xyz[:,0]))


# %%


def sprinkle_observations(obs_xyz, dz=10.0, return_swarm=False, return_index=False):
    """
    Place observations on top boundary wall of the mesh - or pretty close to... (parallel safe)
    """
    inside_particles_g = np.zeros(obs_xyz.shape[0], dtype=np.int32)

    while not inside_particles_g.all():
        swarm_well = uw.swarm.Swarm(mesh=mesh, particleEscape=False)
        particle_index = swarm_well.add_particles_with_coordinates(obs_xyz)

        inside_particles_l = (particle_index >= 0).astype(np.int32)
        inside_particles_g.fill(0)

        uw.mpi.comm.Allreduce([inside_particles_l, MPI.INT], [inside_particles_g, MPI.INT], op=MPI.SUM)

        # print(comm.rank, np.count_nonzero(inside_particles_g == 0))
        obs_xyz[inside_particles_g == 0, 2] -= dz

    output_tuple = [obs_xyz]

    if return_swarm:
        output_tuple.append(swarm_well)
    if return_index:
        output_tuple.append(particle_index)
    return output_tuple


# %%


def reduce_to_root(vals, particle_index):
    """
    Gather values from all processors to the root processor
    """
    nparticles = len(particle_index)

    # initialise with very low numbers
    vl = np.full(nparticles, -999999, np.float32)
    vg = np.full(nparticles, -999999, np.float32)
    vl[particle_index > -1] = vals.ravel()

    # finds the max - aka proper value
    uw.mpi.comm.Reduce([vl, MPI.FLOAT], [vg, MPI.FLOAT], op=MPI.MAX, root=0)
    return vg


# %%


# load recharge data
recharge_data = None
if uw.mpi.rank == 0:
    ti = time()

    recharge_data = pd.read_csv(data_dir+'rch_fnl2_mmyr.csv')


    ### remove data that doesn't fit within the x bounds
    recharge_data = recharge_data[(recharge_data['X'] > xmin) & (recharge_data['X'] < xmax)]
    ### remove data that doesn't fit within the y bounds
    recharge_data = recharge_data[(recharge_data['Y'] > ymin) & (recharge_data['Y'] < ymax)]

    ### remove duplicates if there are any
    recharge_data = recharge_data.drop_duplicates(subset=['X', 'Y'], keep='first')


    # recharge_data = recharge_data[::10]


uw.mpi.comm.barrier()

recharge_data = uw.mpi.comm.bcast(recharge_data, root=0)

uw.mpi.comm.barrier()

recharge_E = recharge_data['X'].values
recharge_N = recharge_data['Y'].values


### convert values from mm/yr to m/s
recharge_vel = ((recharge_data['RechargeRates'].values * u.millimeter/u.year).to(u.meter/u.second).magnitude)

### recharge std based on std of rr increasing as rr increase. Base STD of 5 mm / yr used
recharge_vel_std = (recharge_vel / 4.) + ((0.1 * u.millimeter/u.year).to(u.meter/u.second).magnitude)




recharge_Z = interp_downsampled(np.c_[recharge_E, recharge_N])

recharge_xyz = np.c_[recharge_E, recharge_N, recharge_Z]
recharge_xyz, swarm_recharge, index_recharge = sprinkle_observations(recharge_xyz, dz=10., return_swarm=True, return_index=True)


if uw.mpi.rank == 0 and verbose:
    print("number of recharge observations = {}".format(recharge_xyz.shape[0]))
    print(f"Time to import recharge observations: {time()-ti} seconds")


# %%



# load gw data
gw_data = None

if uw.mpi.rank == 0:

    ti = time()

    gw_data = pd.read_csv(data_dir+"NGIS_groundwater_levels_to_2000_GAB.csv", usecols=(0,3,4,6,7,8,9))


    ### only use data which has a std value above 0
#     gw_data = gw_data[gw_data['gw_level_std'] > 0.1]


    ### remove data not within x bounds
    gw_data = gw_data[(gw_data['easting'] > xmin) & (gw_data['easting'] < xmax)]
    ### remove data not within y bounds
    gw_data = gw_data[(gw_data['northing'] > ymin) & (gw_data['northing'] < ymax)]

    ### remove duplicates if there are any
    gw_data = gw_data.drop_duplicates(subset=['easting', 'northing'], keep='first')

    gw_data.reset_index(inplace=True)

    gw_data['matIndex'] = 0.

    matIndexGWobs = []
    gwObsPoints_data = []
    gwObsPoints_model = []
    pointIndex = []

    topo = topo_interp((gw_data['northing'], gw_data['easting']))
    # topo = interp_downsampled(np.c_[gw_data['easting'].values, gw_data['northing'].values])


    for index in matIndex:
        ### top surface

        row = df_layers.loc[df_layers['mat index'] == index]
        with rioxarray.open_rasterio(surface_filename_tiff.format(row['Layer name'].iloc[0])) as npz:
            layer_interp_top = RegularGridInterpolator((np.flipud(npz.sel(band=1).y.data), npz.sel(band=1).x.data), np.flipud(npz.sel(band=1).data), bounds_error=False)
        ### bottom surface
        row = df_layers.loc[df_layers['mat index'] == index+1]
        with rioxarray.open_rasterio(surface_filename_tiff.format(row['Layer name'].iloc[0])) as npz:
            layer_interp_bottom = RegularGridInterpolator((np.flipud(npz.sel(band=1).y.data), npz.sel(band=1).x.data), np.flipud(npz.sel(band=1).data), bounds_error=False)



        top = layer_interp_top((gw_data['northing'].values, gw_data['easting'].values))
        bottom = layer_interp_bottom((gw_data['northing'].values, gw_data['easting'].values))

#         top_dataPoints = layer_interp_top((y_gw, x_gw))
#         bottom_dataPoints = layer_interp_bottom((y_gw, x_gw))

        matIndexGWobs.append(index)



        gwObsPoints_data.append((gw_data[((topo - gw_data['gw_bore_depth']) < top) & ((topo - gw_data['gw_bore_depth']) > bottom)].shape[0]))



#         gwObsPoints_model.append((gw_xyz[(z_gw < top_dataPoints) & (z_gw > bottom_dataPoints)].shape[0]))


        ### get the index of values that should be in each layer
        pointIndex.append((gw_data[((topo - gw_data['gw_bore_depth']) < top) & ((topo - gw_data['gw_bore_depth']) > bottom)].index.values))


        ### add mat index to df for visualisations
        gw_data.loc[((topo - gw_data['gw_bore_depth']) < top) & ((topo - gw_data['gw_bore_depth']) > bottom), 'matIndex'] = index

    #### covert list to np array
    pointIndex = np.asarray([y for x in pointIndex for y in x])

    ### extract the points for each layer
    gw_data = gw_data[np.isin(gw_data.index.values, pointIndex)]




uw.mpi.comm.barrier()

gw_data = uw.mpi.comm.bcast(gw_data, root=0)

uw.mpi.comm.barrier()


gw_boreID, gw_E, gw_N, gw_elevation, gw_depth, gw_level, gw_level_std = gw_data['ID'], gw_data['easting'].values, gw_data['northing'].values, gw_data['elevation'].values, gw_data['gw_bore_depth'].values, gw_data['gw_level'].values, gw_data['gw_level_std'].values


gw_hydraulic_head = gw_elevation - gw_level
gw_hydraulic_head_std = gw_level_std + 5
gw_pressure_head = gw_depth - gw_level
gw_pressure_head_std = gw_level_std + 5

### get the initial surface elevation
# gw_Z = interp_downsampled(np.c_[gw_E, gw_N])
gw_Z = topo_interp((gw_N, gw_E))

gw_xyz = np.c_[gw_E, gw_N, gw_Z]
# gw_xyz, swarm_gw, index_gw = sprinkle_observations(gw_xyz, dz=10., return_swarm=True, return_index=True)

#### elevation minus the bore depth
gw_xyz[:,2] -= gw_depth

### place observation points in the model
gw_xyz0, swarm_gw0, index_gw0 = sprinkle_observations(gw_xyz, dz=10., return_swarm=True, return_index=True)

topo = topo_interp((gw_data['northing'], gw_data['easting']))

### only use points that are close to original points
gw_xyz = gw_xyz[np.isclose(gw_xyz0[:,2], (topo - gw_data['gw_bore_depth']))]

gw_hydraulic_head = gw_hydraulic_head[np.isclose(gw_xyz0[:,2], (topo - gw_data['gw_bore_depth']))]
gw_hydraulic_head_std = gw_hydraulic_head_std[np.isclose(gw_xyz0[:,2], (topo - gw_data['gw_bore_depth']))]
gw_pressure_head = gw_pressure_head[np.isclose(gw_xyz0[:,2], (topo - gw_data['gw_bore_depth']))]
gw_pressure_head_std = gw_pressure_head_std[np.isclose(gw_xyz0[:,2], (topo - gw_data['gw_bore_depth']))]




uw.mpi.comm.barrier()

if uw.mpi.rank == 0:
    #### save processed gw_data file on root proc
    gw_data = gw_data[np.isclose(gw_xyz0[:,2], (topo - gw_data['gw_bore_depth']))]
    gw_data.to_csv(simulation_directory + 'gw_obs_data.csv')

uw.mpi.comm.barrier()



### place points in the model
gw_xyz, swarm_gw, index_gw = sprinkle_observations(gw_xyz, dz=10., return_swarm=True, return_index=True)


if uw.mpi.rank == 0 and verbose:
    print("number of groundwater pressure observations = {}".format(gw_xyz.shape[0]))
    print(f"Time to import pressure observations: {time()-ti} seconds")



initialgwHydraulicHead = gwHydraulicHead.data[:].copy()



def forward_model(x):
    """
    Variables in x:
    - k_h  : hydraulic conductivity
    - k_t  : thermal conductivity
    - H    : heat production
    - Tmax : bottom temperature BC
    """
    ti = time()

    FM_start = time()

    global niter

    # check we haven't already got a solution
    dist, idx = mintree.query(x)

    if dist == 0.0 and surrogate:
        misfit = minimiser_misfits[idx]
        if verbose:
            print("using surrogate model, misfit = {}".format(misfit))
        return misfit
    else:
        if hydraulicConductivityOnly == True:

            ### scale variables
            kh = x
            global Tmax, kt0, H0
            kt = kt0
            H  = H0*1e6

        else:
        # unpack input vector
            kh, kt, H = np.array_split(x[:-1], 3)
            Tmax = x[-1]

        ### scale variables
        kh = 10.0**kh # log10 scale
        H  = H*1e-6 # convert to micro





        # initialise "default values"
        if hydraulicConductivityOnly == True:
            hydraulicDiffusivity0.data[:]   = kh[-1]
            fn_hydraulicDiffusivity.data[:] = kh[-1]
        else:
            hydraulicDiffusivity0.data[:]   = kh[-1]
            fn_hydraulicDiffusivity.data[:] = kh[-1]

            thermalDiffusivity0.data[:]     = kt[-1]
            heatProduction0.data[:]         = H[-1]




        # populate swarm variables with material properties
        if hydraulicConductivityOnly == True:
            for i in matIndex:
                mask_material = materialIndex0.data == i
                hydraulicDiffusivity0.data[mask_material] = kh[i]

            # determine mean of cells to produce uniform cells
            fn_hydraulicDiffusivity.data[:,0] = np.repeat((np.mean((np.split(hydraulicDiffusivity0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)

        else:
            for i in matIndex:
                mask_material = materialIndex0.data == i
                hydraulicDiffusivity0.data[mask_material] = kh[i]
                thermalDiffusivity0.data[mask_material]   = kt[i]
                heatProduction0.data[mask_material]       = H[i]

            # determine mean of cells to produce uniform cells
            fn_hydraulicDiffusivity.data[:,0] = np.repeat((np.mean((np.split(hydraulicDiffusivity0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)
            thermalDiffusivity.data[:,0] = np.repeat((np.mean((np.split(thermalDiffusivity0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)
            heatProduction.data[:,0] = np.repeat((np.mean((np.split(heatProduction0.data[:,0], np.unique(swarm0.owningCell.data[:,0], return_index = True)[1])[1:]), axis=1)), GPC_solver**mesh.dim)


            ### depth-dependent hydraulic conductivity

            # fn_hydraulicDiffusivity.data[:] = fn_kappa(fn_hydraulicDiffusivity.data.ravel(), depth, beta).reshape(-1,1)


        ### reset velocity and gw hydraulic head values
        velocityField.data[:] = 1e-6

        swarmVelocity.data[:] = 1e-6

        gwHydraulicHead.data[:] = initialgwHydraulicHead




#         print(np.isin(kh, np.unique(hydraulicDiffusivity.data)))





        # # project HP to mesh
        # HPproj = uw.utils.MeshVariable_Projection(heatProductionField, heatProduction, swarm)
        # HPproj.solve()

        GWsolverTimeStart = time()

        ## Set up groundwater equation
        if uw.mpi.rank == 0 and verbose:
            print("Solving grounwater equation...")
        gwsolver.solve()

        GWsolverTimeEnd = time()



        ## calculate velocity from Darcy velocity
        # velocityField.data[:] /= np.clip(fn_porosity(depth_mesh*1e-3, 0.474, 0.071, 5.989), 0.0, 1.0).reshape(-1,1)
        velocityField.data[:] /= porosity.data[:]

        # temperature-dependent conductivity
        temperatureField.data[:] = np.clip(temperatureField.data, Tmin, Tmax)
        temperatureField.data[topWall] = Tmin
        temperatureField.data[bottomWall] = Tmax

        # ## Set up heat equation
        # if uw.mpi.rank == 0 and verbose:
        #     print("Solving heat equation...")
        # for its in range(0, 20):
        #     temperatureField0.data[:] = temperatureField.data[:]
        #     heatsolver.solve(nonLinearIterate=False)
        #
        #     Tdiff = np.array(np.abs(temperatureField0.data[:] - temperatureField.data[:]).max())
        #     Tdiff_all = np.array(0.0)
        #     comm.Allreduce([Tdiff, MPI.DOUBLE], [Tdiff_all, MPI.DOUBLE], op=MPI.MAX)
        #     if Tdiff_all < 0.01:
        #         break


        ### compare to observations and determine misfit

        #         sim_dTdz = temperatureField.fn_gradient[2].evaluate(swarm_dTdz)
        #         sim_dTdz = reduce_to_root(sim_dTdz, index_dTdz)
        #         if uw.mpi.rank == 0:
        #             sim_dTdz = -1.0*sim_dTdz.ravel()
        #             misfit += (((well_dTdz - sim_dTdz)**2/0.1**2).sum())/well_dTdz.size
        #             # print(((well_dTdz - sim_dTdz)**2/0.1**2).sum())

        ### Determine velocity misfit
        sim_vel = uw.function.math.dot(velocityField, velocityField).evaluate(swarm_recharge)
        sim_vel = reduce_to_root(sim_vel, index_recharge)


        ### Determine pressure misfit
        sim_pressure_head = gwHydraulicHead.evaluate(swarm_gw) - zCoordFn.evaluate(swarm_gw)
        sim_pressure_head = reduce_to_root(sim_pressure_head, index_gw)

        def LnormMisfit(p, misfit):
            # global recharge_vel, sim_vel, recharge_vel_std, gw_pressure_head, sim_pressure_head, gw_pressure_head_std, kh, kh0, kt, kt0, dkt, H, H0, dH

            misfitType = f'L{p}-Norm'

            velocity_misfit = (np.abs(np.log10(recharge_vel) - np.log10(sim_vel))**p/np.abs(np.log10(recharge_vel_std))**p).sum() #/ recharge_vel.size

            misfit += velocity_misfit

            pressure_misfit = (np.abs(gw_pressure_head - sim_pressure_head)**p/gw_pressure_head_std**p).sum() # / gw_pressure_head.size

            misfit += pressure_misfit

            ### compare hydraulic conductivity
            HC_misfit = (np.abs(np.log10(kh) - np.log10(kh0))**p).sum()

            misfit += HC_misfit


            ### Compare thermal conductivity
            TC_misfit = (np.abs(kt - kt0)**p/dkt**p).sum()

            misfit += TC_misfit



            ### Compare heat production
            HP_misfit = (np.abs(H - H0)**p/dH**p).sum()

            misfit += HP_misfit

            return misfitType, misfit, velocity_misfit, pressure_misfit, HC_misfit, TC_misfit, HP_misfit

        # misfit = np.array(0.0)
        ### compare priors
        # if uw.mpi.rank == 0:
        #     p_value = 1
        #     misfitType, misfit, velocity_misfit, pressure_misfit, HC_misfit, TC_misfit, HP_misfit  = LnormMisfit(p=p_value, misfit=misfit)
        #
        #     velMisfit.append(velocity_misfit)
        #     pressureMisfit.append(pressure_misfit)
        #     HCMisfit.append(HC_misfit)
        #     totalMisfit.append(misfit)
        #     # iteration.append(niter)
        #
        #     misfitData = pd.DataFrame()
        #     misfitData['iteration'] = misfitData.index.values
        #     misfitData['velMisfit'] = velMisfit
        #     misfitData['pressureMisfit'] = pressureMisfit
        #     misfitData['HCMisfit'] = HCMisfit
        #     misfitData['totalMisfit'] = totalMisfit
        #
        #     misfitData.to_csv(simulation_directory + str(misfitType) + '-misfitdata.csv')
        #
        #     if verbose == True:
        #         print(f"Misfit Type: {misfitType}")
        #         print(f'Velocity misfit: {velocity_misfit}')
        #         print(f'Pressure misfit: {pressure_misfit}')
        #         print(f'Hydraulic conductivity misfit: {HC_misfit}')
        #         print(f'Thermal conductivity misfit: {TC_misfit}')
        #         print(f'Heat production misfit: {HP_misfit}')
        #
        #         print(f"Total misfit: {misfit}")

        FM_end = time()

        FM_time = FM_end - FM_start

        GWTime = GWsolverTimeEnd - GWsolverTimeStart

        misfit = np.array(0.0)
        ### compare priors
        if uw.mpi.rank == 0:
            p_value = 2
            misfitType, misfit, velocity_misfit, pressure_misfit, HC_misfit, TC_misfit, HP_misfit  = LnormMisfit(p=p_value, misfit=misfit)

            velMisfit.append(velocity_misfit)
            pressureMisfit.append(pressure_misfit)
            HCMisfit.append(HC_misfit)
            totalMisfit.append(misfit)
            FMTime.append(FM_time)
            GWsolverTime.append(GWTime)
            # iteration.append(niter)

            if niter < BURNIN:
                burninPhase.append(1)
            else:
                burninPhase.append(0)


            misfitData = pd.DataFrame()

            misfitData['Burnin phase'] = burninPhase
            misfitData['velMisfit'] = velMisfit
            misfitData['pressureMisfit'] = pressureMisfit

            misfitData['HCMisfit'] = HCMisfit
            misfitData['totalMisfit'] = totalMisfit
            misfitData['GWSolverTime'] = GWsolverTime
            misfitData['Time'] = FMTime
            misfitData.insert(loc=0, column='iteration', value=np.arange(0,len(FMTime)))

            misfitData.to_csv(simulation_directory + str(misfitType) + '-misfitdata.csv')

            if verbose == True:
                print(f"Misfit Type: {misfitType}")
                print(f'Velocity misfit: {velocity_misfit}')
                print(f'Pressure misfit: {pressure_misfit}')
                print(f'Hydraulic conductivity misfit: {HC_misfit}')
                print(f'Thermal conductivity misfit: {TC_misfit}')
                print(f'Heat production misfit: {HP_misfit}')

                print(f"Total misfit: {misfit}")






        comm.Bcast([misfit, MPI.DOUBLE], root=0)



        ### in MH model, discard BURNIN results
        if n_checkpoints == 1 and niter % n_checkpoints == 0 and niter > BURNIN:
    #         # temperatureField.save(data_dir+'checkpoints/temperatureField_{:06d}.h5'.format(niter))
            gwHydraulicHead.save(simulation_directory+'checkpoints/hydraulicHeadField_{:06d}.h5'.format(niter))
            velocityField.save(simulation_directory+'checkpoints/velocityField_{:06d}.h5'.format(niter))
        #         # if uw.mpi.rank == 0:
        #             # np.savetxt(simulation_directory + 'kh_{:06d}.txt'.format(niter), kh, delimiter=',')
        #
        #
        #### save all results to show how the burnin phase does too
        if uw.mpi.rank == 0:
            with open(simulation_directory+'minimiser_results.csv', 'a') as f:
                rowwriter = csv.writer(f, delimiter=',')
                rowwriter.writerow(np.hstack([[misfit], x]))

            if verbose:
                print("\n rank {} in {:.2f} sec misfit = {}\n".format(uw.mpi.rank, time()-ti, misfit))

        niter += 1

        return misfit


# %%
def metropolis_hastings(func, x0, nsim, burnin, x_scale=(1,1), tempering=10e3, ):
    """
    MCMC algorithm using a Metropolis-Hastings sampler.
    Evaluates a Markov-Chain for starting values of
    \\( \\beta, z_t, \\Delta z, C \\) and returns the
    ensemble of model realisations.

    Args:
        nsim : int
            number of simulations
        burnin : int
            number of burn-in simulations before to nsim
        x_scale: float(4) (optional)
            scaling factor for new proposals
            (default=`[1,1,1,1]` for `[beta, zt, dz, C]`)
            - see notes
    Returns:
        X
    """
    x0 = np.array(x0)
    size = len(x_scale)
    samples = np.empty((nsim, size))
    acceptance = np.zeros((nsim, 1))
    accept_rate = np.empty((nsim, 1))
    misfit_values = np.empty((nsim, 1))

    misfit0 = func(x0)
    P0 = np.exp(-misfit0 / tempering)

    # Burn-in phase
    for i in range(burnin):
        # add random perturbation

        x1 = None

        if uw.mpi.rank == 0:

            x1 = x0 + np.random.normal(size=size) * x_scale

        uw.mpi.comm.barrier()

        x1 = uw.mpi.comm.bcast(x1, root=0)

        uw.mpi.comm.barrier()

        # evaluate proposal probability + tempering
        misfit1 = func(x1)
        P1 = np.exp(-misfit1 / tempering)

        # iterate towards MAP estimate
        if P1 > P0:
            x0 = x1
            P0 = P1
            misfit0 = misfit1

    # misfit0 = func(x0)
    # P0 = np.exp(-misfit0 / 10e3)

    naccept = 0

    # Now sample posterior
    for i in range(nsim):

        # add random perturbation
        x1 = None

        if uw.mpi.rank == 0:

            x1 = x0 + np.random.normal(size=size) * x_scale

        uw.mpi.comm.barrier()

        x1 = uw.mpi.comm.bcast(x1, root=0)

        uw.mpi.comm.barrier()

        # evaluate proposal probability
        P0 = max(P0, 1e-99)

        misfit1 = func(x1)
        P1 = np.exp(-misfit1 / tempering)

        P = min(P1 / P0, 1.0)

        # randomly accept probability
        randProbability = None

        if uw.mpi.rank == 0:

            randProbability = np.random.rand()

        uw.mpi.comm.barrier()

        randProbability = uw.mpi.comm.bcast(randProbability, root=0)

        uw.mpi.comm.barrier()


        if randProbability <= P:
            x0 = x1
            P0 = P1
            misfit0 = misfit1
            acceptance[i] = 1
            naccept += 1



        misfit_values[i] = misfit0
        samples[i] = x0
        accept_rate[i] = naccept / i

    return samples, acceptance, misfit_values, accept_rate


# %%




if hydraulicConductivityOnly == True:
    x = np.hstack([np.log10(kh0)])
    # define bounded optimisation
    bounds_lower = np.hstack([
        np.full_like(kh0, -15)])
    bounds_upper = np.hstack([
        np.full_like(kh0, -3)])
    bounds = list(zip(bounds_lower, bounds_upper))
else:
    x = np.hstack([np.log10(kh0), kt0, H0*1e6, [Tmax]])
    # define bounded optimisation
    bounds_lower = np.hstack([
        np.full_like(kh0, -15),
        np.full_like(kt0, 0.05),
        np.zeros_like(H0),
        [298.]])
    bounds_upper = np.hstack([
        np.full_like(kh0, -3),
        np.full_like(kt0, 6.0),
        np.full_like(H0, 10),
        [600+273.14]])

    bounds = list(zip(bounds_lower, bounds_upper))


dx = 0.01*x
# -

# ## Initialise output table
#
# A place to store misfit and $x$ parameters.

# +
import os

if "minimiser_results.csv" in os.listdir(simulation_directory):
    # load existing minimiser results table
    minimiser_results_data = np.loadtxt(simulation_directory+"minimiser_results.csv", delimiter=',', )
    if not len(minimiser_results_data):
        minimiser_results_data = np.zeros((1,x.size+1))
    minimiser_results = minimiser_results_data[:,1:]
    minimiser_misfits = minimiser_results_data[:,0]
else:
    minimiser_results = np.zeros((1,x.size))
    minimiser_misfits = np.array([0.0])
    if uw.mpi.rank == 0:
        with open(simulation_directory+'minimiser_results.csv', 'w') as f:
            pass

mintree = cKDTree(minimiser_results)








# finite_diff_step = np.hstack([np.full_like(kh0, 0.1), np.full_like(kt0, 0.01), np.full_like(H0, 0.01), [1.0]])

# def obj_func(x):
#     return forward_model(x)
# def obj_grad(x):
#     return optimize.approx_fprime(x, forward_model, finite_diff_step)


# %%

velMisfit      = []
pressureMisfit = []
totalMisfit    = []
# iteration      = []
HCMisfit       = []
FMTime         = []
GWsolverTime   = []
burninPhase    = []


niter = 0

# +
#### test forward model
fm0 = forward_model(x)
# fm1 = forward_model(x+dx)
# print("finite difference = {}".format(fm1-fm0))


#
#
## Check gradient of variables in the forward model
# if hydraulicConductivityOnly == True:
#     finite_diff_step = np.hstack([np.full_like(kh0, 1.)])
# else:
#     finite_diff_step = np.hstack([np.full_like(kh0, 1.), np.full_like(kt0, 0.01), np.full_like(H0, 1.), [1.0]])
#
# fprime_data = optimize.approx_fprime(xk=x, f=forward_model, epsilon=finite_diff_step)
#
# if uw.mpi.rank == 0:
#     print(f'fprime data: {fprime_data}')
#
#     fprime_df = pd.DataFrame()
#     fprime_df['fprime data'] = fprime_data
#     fprime_df.to_csv(simulation_directory + 'fprime_data.csv')


# %%



# ### differential evolution
# res = optimize.differential_evolution(func=forward_model, bounds=bounds, args=(niter,), popsize=2, seed=42, disp=True, x0=x)


# shgo_kwargs = dict(method='L-BFGS-B')

# res = optimize.shgo(forward_model, bounds=bounds, args=(niter,), minimizer_kwargs=shgo_kwargs)

# basinhopping_kwargs = dict(bounds=bounds, method = "L-BFGS-B", args =(niter,))
# res = optimize.basinhopping(forward_model, x0=x, minimizer_kwargs=basinhopping_kwargs, seed=42, niter=1000)




# # ### metropolis hastings
# samples, acceptance, misfit_values, accept_rate = metropolis_hastings(func=forward_model, x0=x, nsim=NSIM, burnin=BURNIN, x_scale=sigma*np.ones_like(x), tempering=10e3, )

# if uw.mpi.rank ==0:
#     MH_results = pd.DataFrame()
#     ### loop over matIndex to create a df with all the samples
#     for i in matIndex:
#         MH_results[str(i)] = samples[:,i]

#     MH_results['acceptance'] = acceptance
#     MH_results['acceptance rate'] = accept_rate
#     MH_results['misfit'] = misfit_values

#     MH_results.to_csv(simulation_directory + 'MH_output.csv')




# print(res)
#
#
#
#

saveDataStart = time()



# if uw.mpi.rank == 0:
#     np.savez_compressed(data_dir+"optimisation_result.npz", **res)


# Save data outputs

# %%


xdmf_info_mesh  = mesh.save(simulation_directory+'mesh.h5')
xdmf_info_swarm = swarm.save(simulation_directory+'swarm.h5')

xdmf_info_matIndex = materialIndex.save(simulation_directory+'materialIndex.h5')
materialIndex.xdmf(simulation_directory+'materialIndex.xdmf', xdmf_info_matIndex, 'materialIndex', xdmf_info_swarm, 'TheSwarm')


# dummy mesh variable
phiField        = mesh.add_variable( nodeDofCount=1 )
heatflowField   = mesh.add_variable( nodeDofCount=3 )


# calculate heat flux
thermalDiffusivity.data[:] = fn_thermalDiffusivity.evaluate(swarm)
kTproj = uw.utils.MeshVariable_Projection(phiField, thermalDiffusivity, swarm)
kTproj.solve()

heatflowField.data[:] = temperatureField.fn_gradient.evaluate(mesh) * -phiField.data.reshape(-1,1)

rankField = mesh.add_variable( nodeDofCount=1 )
rankField.data[:] = uw.mpi.rank

pressureField = gwHydraulicHead.copy(deepcopy=True)
pressureField.data[:] -= zCoordFn.evaluate(mesh)


for xdmf_info,save_name,save_object in [(xdmf_info_mesh, 'velocityField', velocityField),
                                        (xdmf_info_mesh, 'hydraulicHeadField', gwHydraulicHead),
                                        (xdmf_info_mesh, 'pressureField', pressureField),
                                        (xdmf_info_mesh, 'temperatureField', temperatureField),
                                        # (xdmf_info_mesh, 'heatflowField', heatflowField),
                                        (xdmf_info_mesh, 'rankField', rankField),
                                        (xdmf_info_mesh, 'materialMesh', materialMesh),
                                        (xdmf_info_swarm, 'materialIndexSwarm', materialIndex),
                                        (xdmf_info_swarm, 'hydraulicDiffusivitySwarm', fn_hydraulicDiffusivity),
                                        # (xdmf_info_swarm, 'thermalDiffusivitySwarm', thermalDiffusivity),
                                        # (xdmf_info_swarm, 'heatProductionSwarm', heatProduction),
                                        ]:

    xdmf_info_var = save_object.save(simulation_directory+save_name+'.h5')
    save_object.xdmf(simulation_directory+save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')

    if save_name.endswith("Swarm"):
        # project swarm variables to the mesh
        hydproj = uw.utils.MeshVariable_Projection(phiField, save_object, swarm)
        hydproj.solve()

        field_name = save_name[:-5]+'Field'
        xdmf_info_var = phiField.save(simulation_directory+field_name+'.h5')
        phiField.xdmf(simulation_directory+field_name+'.xdmf', xdmf_info_var, field_name, xdmf_info_mesh, "TheMesh")

# +
# xdmf_info_swarm_dTdz     = swarm_dTdz.save(data_dir+'swarm_dTdz.h5')
xdmf_info_swarm_recharge = swarm_recharge.save(simulation_directory+'swarm_recharge.h5')
xdmf_info_swarm_gw       = swarm_gw.save(simulation_directory+'swarm_gw.h5')

# interpolate to swarm variables (again)
# sim_dTdz = temperatureField.fn_gradient[2].evaluate(swarm_dTdz)
sim_vel = uw.function.math.dot(velocityField, velocityField).evaluate(swarm_recharge)
sim_pressure_head = gwHydraulicHead.evaluate(swarm_gw) - zCoordFn.evaluate(swarm_gw)


for save_name, this_swarm, swarm_obs, swarm_sim, index_field in [
#         ('dTdz', swarm_dTdz, well_dTdz, sim_dTdz, index_dTdz),
        ('recharge', swarm_recharge, recharge_vel, sim_vel, index_recharge),
        ('pressure_head', swarm_gw, gw_pressure_head, sim_pressure_head, index_gw)]:

    xdmf_info_this_swarm = this_swarm.save(simulation_directory+'swarm_{}.h5'.format(save_name))

    # save obs
    swarm_obs_var = this_swarm.add_variable( dataType="double", count=1 )
    swarm_obs_var.data[:] = swarm_obs[index_field > -1].reshape(-1,1)
    xdmf_info_var = swarm_obs_var.save(simulation_directory+'obs_'+save_name+'.h5')
    swarm_obs_var.xdmf(simulation_directory+'obs_'+save_name+'.xdmf', xdmf_info_var, save_name,
                         xdmf_info_this_swarm, 'swarm_{}.h5'.format(save_name))

    # save sim
    swarm_sim_var = this_swarm.add_variable( dataType="double", count=1 )
    swarm_sim_var.data[:] = swarm_sim.reshape(-1,1)
    xdmf_info_var = swarm_sim_var.save(simulation_directory+'sim_'+save_name+'.h5')
    swarm_sim_var.xdmf(simulation_directory+'sim_'+save_name+'.xdmf', xdmf_info_var, save_name,
                         xdmf_info_this_swarm, 'swarm_{}.h5'.format(save_name))
# -

# ## Save minimiser results





# %%
endTiming = time()

if uw.mpi.rank == 0:
    scriptTiming = endTiming - startTiming
    saveDataTiming = endTiming - saveDataStart
    solverTiming = sum(FMTime) / len(FMTime)


    timingData = np.zeros((1, 6))
    timingData[:,0] = (Nx*Ny*Nz)
    timingData[:,1] = (GPC_solver**3)
    timingData[:,2] = (Nx*Ny*Nz*(GPC_solver**3))
    timingData[:,3] = saveDataTiming
    timingData[:,4] = solverTiming
    timingData[:,5] = scriptTiming


    ### save results to df in sim directory
    timing_data = pd.DataFrame()


    timing_data['totalCells'] = timingData[:,0]
    timing_data['particlesPerCell'] = timingData[:,1]
    timing_data['totalParticles'] = timingData[:,2]
    timing_data['scriptTime'] = timingData[:,5]
    timing_data['solverTime'] = timingData[:,4]
    timing_data['saveDataTime'] = timingData[:,3]

    timing_data.to_csv(simulation_directory + 'timing_data.csv')

# %%
