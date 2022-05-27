# %% [markdown]
#  ### code to calculate summary stats of model
#  p10, p50, p90, mean, mode, standard deviation

# %%
import numpy as np
import os
import pandas as pd
from scipy import stats

import fnmatch

import underworld as uw


# %%
### define path files are stored in

### where the required files are stored
path = r'/scratch/q97/bk2562/ensembleData/'


### where velocity field is stored
checkpoints = r'/scratch/q97/bk2562/checkpoints/'

### where to save the esemble stats
output = r'/scratch/q97/bk2562/ensembleResults/'

if uw.mpi.rank == 0:
    if not os.path.exists(output):
        os.makedirs(output)


# ### where the required files are stored
# path = r'/Volumes/Seagate BarraCuda 120/GroundwaterProject-GA/Models/forwardModel/newkh0Values+MHModel-Tuned-50Burnin_500Nsims_10.0dx-10.0dy-0.1dz_8PPC/ensembleData/'

# ### where velocity field is stored
# checkpoints = r'/Volumes/Seagate BarraCuda 120/GroundwaterProject-GA/Models/forwardModel/newkh0Values+MHModel-Tuned-50Burnin_500Nsims_10.0dx-10.0dy-0.1dz_8PPC/checkpoints/'




# %%
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


    ### count duplicates in list. Calculates how many times to repeat file if it appears multiple times in the ensemble 

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



# %%
# define bounding box
xmin, xmax, ymin, ymax = -955637.8812, 1034362.2443650428, 6342298.2975, 8922298.39436168
zmin, zmax = -8000.0, 1200.0

# resolution
dx, dy, dz = 10e3, 10e3, 100
# dx, dy, dz = 10e3, 10e3, 50
Nx, Ny, Nz = int((xmax-xmin)/dx), int((ymax-ymin)/dy), int((zmax-zmin)/dz)



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
mesh.load(checkpoints+'mesh.h5')

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
vx = np.zeros((dummyField.data.shape[0], its))
vy = np.zeros((dummyField.data.shape[0], its))
vz = np.zeros((dummyField.data.shape[0], its))
v = np.zeros((dummyField.data.shape[0], its))

if uw.mpi.rank == 0:
    print('created numpy arrays')
    print(vx.shape)


# %%
### a for number of reps for each it
a = 0
### j for column in np array to update
j = 0
### loops over each file, reading in the velocity data. Sorts file list to make sure each cpu does the same file order
for file in sorted(os.listdir(checkpoints)[:]):
    ### only use velocityField files
    if fnmatch.fnmatch(file, 'velocityField*'):
        #### opens the file
        velocityField.load(checkpoints + file )
#         print('loaded in file')
        ### loop over file to import it multiple times if value is repeated
        for x in range(0, reps[a]):
            vx[:,j] = velocityField.data[:,0]
            vy[:,j] = velocityField.data[:,1]
            vz[:,j] = velocityField.data[:,2]
            v[:,j]  = np.linalg.norm(velocityField.data[:,], axis=1)

            j +=1

        if uw.mpi.rank ==0:
            print(file + ' imported')
        a+= 1

        ### wait for all procs before going onto the next file
        uw.mpi.comm.barrier()

if uw.mpi.rank ==0:
    print('j value to verify ensemble: ' + str(j))


if uw.mpi.rank == 0:
    print('loaded in ensemble')


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
p50VelocityField = mesh.add_variable( nodeDofCount=3 )

p50VelocityField.data[:,0] = np.percentile(a=vx, q=50, axis=1)
p50VelocityField.data[:,1] = np.percentile(a=vy, q=50, axis=1)
p50VelocityField.data[:,2] = np.percentile(a=vz, q=50, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p50')


# %%
p90VelocityField = mesh.add_variable( nodeDofCount=3 )

p90VelocityField.data[:,0] = np.percentile(a=vx, q=90, axis=1)
p90VelocityField.data[:,1] = np.percentile(a=vy, q=90, axis=1)
p90VelocityField.data[:,2] = np.percentile(a=vz, q=90, axis=1)

if uw.mpi.rank == 0:
    print('finished calculating p90')


# %%
magVelocityField = mesh.add_variable( nodeDofCount=6 )


magVelocityField.data[:,0] = v.mean(axis=1)
magVelocityField.data[:,1] = v.std(axis=1)
magVelocityField.data[:,2] = stats.mode(v, axis=1)[0][:,0]
magVelocityField.data[:,3] = np.percentile(a=v, q=10, axis=1)
magVelocityField.data[:,4] = np.percentile(a=v, q=50, axis=1)
magVelocityField.data[:,5] = np.percentile(a=v, q=90, axis=1)

if uw.mpi.rank == 0:
    print('finished saving mag')



# %%
Fields = [meanVelocityField, stdVelocityField, modeVelocityField, p10VelocityField, p50VelocityField, p90VelocityField,magVelocityField]





# %%
xdmf_info_mesh  = mesh.save(output+'mesh.h5')



for xdmf_info,save_name,save_object in [(xdmf_info_mesh, 'meanVelocityField', meanVelocityField),
                                        (xdmf_info_mesh, 'stdVelocityField', stdVelocityField),
                                        (xdmf_info_mesh, 'modeVelocityField', modeVelocityField),
                                        (xdmf_info_mesh, 'p10VelocityField', p10VelocityField),
                                        (xdmf_info_mesh, 'p50VelocityField', p50VelocityField),
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
                                        (xdmf_info_mesh1, 'p50VelocityField_newZ', p50VelocityField),
                                        (xdmf_info_mesh1, 'p90VelocityField_newZ', p90VelocityField),
                                        (xdmf_info_mesh1, 'magVelocityField_newZ', magVelocityField),
                                        ]:

    xdmf_info_var = save_object.save(output+save_name+'.h5')
    save_object.xdmf(output+save_name+'.xdmf', xdmf_info_var, save_name, xdmf_info, 'TheMesh')


if uw.mpi.rank == 0:
    print('finished saving new Z fields to h5')



# %%



# %%
