# %% [markdown]

### code to calculate summary stats of model
### p10, p25, p50, p75, p90, mean, mode, standard deviation
'''
MLE and MAP fields have to be updated manually
'''
# %%
### calc stats
import numpy as np
import pandas as pd
from scipy import stats

### directories and file structures
import os
import sys
import fnmatch
import underworld as uw
from natsort import natsorted
import glob

### load and create datasets
import pyvista as pv
import h5py


# %%
### define path files are stored in

### where the required files are stored

data_dir = r'/scratch/q97/bk2562/GAB_MHmodel_test/GAB_forwardModel/'


### where velocity field is stored
checkpoints = data_dir + r'checkpoints/'

### where to save the esemble stats
output = data_dir + r'ensembleResults/'

if uw.mpi.rank == 0:
    if not os.path.exists(output):
        os.makedirs(output)



# %%
reps = None
its  = None

if uw.mpi.rank == 0 :

    #### load in the datasts
    complete_set = pd.read_csv(data_dir + 'MH_output.csv', index_col=[0])[4:]



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


if its != sum(reps):
    sys.exit("Sum of reps does not equal number of iterations")





# %%
# define bounding box
xmin, xmax, ymin, ymax = -955637.8812, 1034362.2443650428, 6342298.2975, 8922298.39436168
zmin, zmax = -8000.0, 1200.0

# resolution
dx, dy, dz = 10e3, 10e3, 100
# dx, dy, dz = 10e3, 10e3, 50
Nx, Ny, Nz = int((xmax-xmin)/dx), int((ymax-ymin)/dy), int((zmax-zmin)/dz)


#
Xcoords = np.linspace(xmin,xmax,Nx)
Ycoords = np.linspace(ymin,ymax,Ny)


# %%
deformedmesh = True
elementType = "Q1"
mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType),
                                 elementRes  = (Nx,Ny,Nz),
                                 minCoord    = (xmin,ymin,zmin),
                                 maxCoord    = (xmax,ymax,zmax))


# %%
mesh.load(data_dir+'mesh.h5')

if uw.mpi.rank == 0:
    print('loaded in mesh')


# %%
velocityField              = mesh.add_variable( nodeDofCount=3 )


# %%
dummyField                 = mesh.add_variable( nodeDofCount=1 )

if uw.mpi.rank == 0:
    print('created UW fields')


# %%
### create arrays with same shape as the dummyField to split the datasets up in parallel
vx = np.zeros((dummyField.data.shape[0], its))*np.nan
vy = np.zeros((dummyField.data.shape[0], its))*np.nan
vz = np.zeros((dummyField.data.shape[0], its))*np.nan
v = np.zeros((dummyField.data.shape[0], its))*np.nan

if uw.mpi.rank == 0:
    print('created numpy arrays')
    print(vx.shape)


# %%
### a for number of reps for each it
a = 0
### j for column in np array to update
j = 0
### loops over each file, reading in the velocity data. Sorts file list to make sure each cpu does the same file order
# for file in sorted(os.listdir(checkpoints)[:]):
#     ### only use velocityField files
#     if fnmatch.fnmatch(file, 'velocityField*'):
    #### opens the file
    # velocityField.load(file)
    #     #         print('loaded in file')
    #     ### loop over file to import it multiple times if value is repeated
        # for x in range(0, reps[a]):
        #     vx[:,j] = velocityField.data[:,0]
        #     vy[:,j] = velocityField.data[:,1]
        #     vz[:,j] = velocityField.data[:,2]
        #     v[:,j]  = np.linalg.norm(velocityField.data[:,], axis=1)
        #
        #     j +=1

for file in natsorted(glob.glob(checkpoints + 'velocityField*')):
    #### opens the file
    velocityField.load(file)
        #         print('loaded in file')
        ### loop over file to import it multiple times if value is repeated

    start = j
    end   = j+reps[a]

    vx[:,start:end] = np.transpose([velocityField.data[:,0]] * reps[a])
    vy[:,start:end] = np.transpose([velocityField.data[:,1]] * reps[a])
    vz[:,start:end] = np.transpose([velocityField.data[:,2]] * reps[a])
    v[:,start:end]  = np.transpose([np.linalg.norm(velocityField.data[:,], axis=1)] * reps[a])

    if uw.mpi.rank ==0:
        print(file + ' imported')

    j+=reps[a]
    a+= 1


    ### wait for all procs before going onto the next file
    uw.mpi.comm.barrier()

if uw.mpi.rank ==0:
    print('j value to verify ensemble: ' + str(j))


if uw.mpi.rank == 0:
    print('loaded in ensemble')

'''check if ensemble arrays contain a NaN value '''

if (np.isnan(vx).any()):
    sys.exit("NaN detected in vx")
if (np.isnan(vy).any()):
    sys.exit("NaN detected in vy")
if (np.isnan(vz).any()):
    sys.exit("NaN detected in vz")
if (np.isnan(v).any()):
    sys.exit("NaN detected in v")


# %%
meanVelocityField = mesh.add_variable( nodeDofCount=3 )

meanVelocityField.data[:,0] = vx.mean(axis=1)
meanVelocityField.data[:,1] = vy.mean(axis=1)
meanVelocityField.data[:,2] = vz.mean(axis=1)

if uw.mpi.rank == 0:
    print('finished calculating mean')


# %%
stdVelocityField = mesh.add_variable( nodeDofCount=3 )

stdVelocityField.data[:,0] = vx.std(axis=1)
stdVelocityField.data[:,1] = vy.std(axis=1)
stdVelocityField.data[:,2] = vz.std(axis=1)

if uw.mpi.rank == 0:
    print('finished calculating std')


# %%
modeVelocityField = mesh.add_variable( nodeDofCount=3 )

modeVelocityField.data[:,0] = stats.mode(vx, axis=1)[0][:,0]
modeVelocityField.data[:,1] = stats.mode(vy, axis=1)[0][:,0]
modeVelocityField.data[:,2] = stats.mode(vz, axis=1)[0][:,0]

if uw.mpi.rank == 0:
    print('finished calculating mode')


# %%
p10VelocityField = mesh.add_variable( nodeDofCount=3 )

p10VelocityField.data[:,0] = np.percentile(a=vx, q=10, axis=1)
p10VelocityField.data[:,1] = np.percentile(a=vy, q=10, axis=1)
p10VelocityField.data[:,2] = np.percentile(a=vz, q=10, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p10')

# %%
p25VelocityField = mesh.add_variable( nodeDofCount=3 )

p25VelocityField.data[:,0] = np.percentile(a=vx, q=25, axis=1)
p25VelocityField.data[:,1] = np.percentile(a=vy, q=25, axis=1)
p25VelocityField.data[:,2] = np.percentile(a=vz, q=25, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p25')


# %%
p50VelocityField = mesh.add_variable( nodeDofCount=3 )

p50VelocityField.data[:,0] = np.percentile(a=vx, q=50, axis=1)
p50VelocityField.data[:,1] = np.percentile(a=vy, q=50, axis=1)
p50VelocityField.data[:,2] = np.percentile(a=vz, q=50, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p50')

# %%
p75VelocityField = mesh.add_variable( nodeDofCount=3 )

p75VelocityField.data[:,0] = np.percentile(a=vx, q=75, axis=1)
p75VelocityField.data[:,1] = np.percentile(a=vy, q=75, axis=1)
p75VelocityField.data[:,2] = np.percentile(a=vz, q=75, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p75')


# %%
p90VelocityField = mesh.add_variable( nodeDofCount=3 )

p90VelocityField.data[:,0] = np.percentile(a=vx, q=90, axis=1)
p90VelocityField.data[:,1] = np.percentile(a=vy, q=90, axis=1)
p90VelocityField.data[:,2] = np.percentile(a=vz, q=90, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p90')


# %%
magVelocityField = mesh.add_variable( nodeDofCount=9 )


magVelocityField.data[:,0] = v.mean(axis=1)
magVelocityField.data[:,1] = v.std(axis=1)
magVelocityField.data[:,2] = stats.mode(v, axis=1)[0][:,0]
magVelocityField.data[:,3] = np.percentile(a=v, q=10, axis=1)
magVelocityField.data[:,4] = np.percentile(a=v, q=25, axis=1)
magVelocityField.data[:,5] = np.percentile(a=v, q=50, axis=1)
magVelocityField.data[:,6] = np.percentile(a=v, q=75, axis=1)
magVelocityField.data[:,7] = np.percentile(a=v, q=90, axis=1)

magVelocityField.data[:,8] = (v.std(axis=1)) / (v.mean(axis=1))

if uw.mpi.rank == 0:
    print('finished saving mag')



# %%
fieldNames = ['meanVelocityField', 'stdVelocityField', 'modeVelocityField', 'p10VelocityField', 'p25VelocityField', 'p50VelocityField', 'p75VelocityField', 'p90VelocityField', 'magVelocityField']
Fields = [meanVelocityField, stdVelocityField, modeVelocityField, p10VelocityField, p25VelocityField, p50VelocityField, p75VelocityField, p90VelocityField,magVelocityField]





# %%
xdmf_info_mesh  = mesh.save(output+'mesh.h5')



for xdmf_info,save_name,save_object in [(xdmf_info_mesh, 'meanVelocityField', meanVelocityField),
                                        (xdmf_info_mesh, 'stdVelocityField', stdVelocityField),
                                        (xdmf_info_mesh, 'modeVelocityField', modeVelocityField),
                                        (xdmf_info_mesh, 'p10VelocityField', p10VelocityField),
                                        (xdmf_info_mesh, 'p25VelocityField', p25VelocityField),
                                        (xdmf_info_mesh, 'p50VelocityField', p50VelocityField),
                                        (xdmf_info_mesh, 'p75VelocityField', p75VelocityField),
                                        (xdmf_info_mesh, 'p90VelocityField', p90VelocityField),
                                        (xdmf_info_mesh, 'magVelocityField', magVelocityField),
                                        ]:

    xdmf_info_var = save_object.save(output+save_name+'.h5')
    save_object.xdmf(output+save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')

if uw.mpi.rank == 0:
    print('finished saving fields to h5')


# %%
#### new z scale for visualisation only
with mesh.deform_mesh():
    mesh.data[:,2] = mesh.data[:,2] * 50

xdmf_info_mesh1  = mesh.save(output+'mesh_newZ.h5')



for xdmf_info,save_name,save_object in [(xdmf_info_mesh1, 'meanVelocityField_newZ', meanVelocityField),
                                        (xdmf_info_mesh1, 'stdVelocityField_newZ', stdVelocityField),
                                        (xdmf_info_mesh1, 'modeVelocityField_newZ', modeVelocityField),
                                        (xdmf_info_mesh1, 'p10VelocityField_newZ', p10VelocityField),
                                        (xdmf_info_mesh1, 'p25VelocityField_newZ', p25VelocityField),
                                        (xdmf_info_mesh1, 'p50VelocityField_newZ', p50VelocityField),
                                        (xdmf_info_mesh1, 'p75VelocityField_newZ', p75VelocityField),
                                        (xdmf_info_mesh1, 'p90VelocityField_newZ', p90VelocityField),
                                        (xdmf_info_mesh1, 'magVelocityField_newZ', magVelocityField),
                                        ]:

    xdmf_info_var = save_object.save(output+save_name+'.h5')
    save_object.xdmf(output+save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')


if uw.mpi.rank == 0:
    print('finished saving new Z fields to h5')

if uw.mpi.rank == 0:
    #### save to vtu using pyvista
    grid = pv.StructuredGrid()

    grid.points = h5py.File(data_dir+'mesh.h5')['vertices'][:]

    grid.dimensions = Nx+1, Ny+1, Nz+1

    grid["materialMesh"] = h5py.File(data_dir + 'materialMesh.h5', 'r')['data'][:]

    grid["MLEvelocityField"] = h5py.File(checkpoints + 'velocityField_000833.h5', 'r')['data'][:]
    grid["MAPvelocityField"] = h5py.File(checkpoints + 'velocityField_001509.h5', 'r')['data'][:]

    fieldNames = ['meanVelocityField', 'stdVelocityField', 'modeVelocityField', 'p10VelocityField', 'p25VelocityField', 'p50VelocityField', 'p75VelocityField', 'p90VelocityField', 'magVelocityField']

    for fieldName in fieldNames:
        grid[fieldName] = h5py.File(output+fieldName+'.h5')['data'][:]

    #### std over mean
    grid["COV"] = (np.log10(grid['magVelocityField'][:,1]))/(np.log10(grid['magVelocityField'][:,0]))


    grid.save(output+'GAB_velocityFields.vtk')








# %%



# %%
