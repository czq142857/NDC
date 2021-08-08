import numpy as np
import h5py


def read_sdf_file_as_3d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#sdf'):
        raise IOError('Not a sdf file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape(dims)
    fp.close()
    return data


def read_data_input_only(hdf5_dir,grid_size,input_type,out_bool,out_float):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    if out_bool:
        LOD_gt_int = np.zeros([grid_size+1,grid_size+1,grid_size+1,1],np.int32)
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,3],np.float32)
    else:
        LOD_gt_float = None
    if input_type=="sdf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    hdf5_file.close()
    return LOD_gt_int, LOD_gt_float, LOD_input

def read_data_bool_only(hdf5_dir,grid_size,input_type,out_bool,out_float):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    if out_bool:
        LOD_gt_int = hdf5_file[str(grid_size)+"_int"][:]
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,3],np.float32)
    else:
        LOD_gt_float = None
    if input_type=="sdf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    hdf5_file.close()
    return LOD_gt_int, LOD_gt_float, LOD_input

def read_data(hdf5_dir,grid_size,input_type,out_bool,out_float):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    if out_bool:
        LOD_gt_int = hdf5_file[str(grid_size)+"_int"][:]
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = hdf5_file[str(grid_size)+"_float"][:]
    else:
        LOD_gt_float = None
    if input_type=="sdf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    hdf5_file.close()
    return LOD_gt_int, LOD_gt_float, LOD_input

def read_and_augment_data(hdf5_dir,grid_size,input_type,out_bool,out_float,aug_permutation=True,aug_reversal=True,aug_inversion=True):
    grid_size_1 = grid_size+1

    #read input hdf5
    LOD_gt_int, LOD_gt_float, LOD_input = read_data(hdf5_dir,grid_size,input_type,out_bool,out_float)

    newdict = {}

    if out_bool:
        newdict['int_V_signs'] = LOD_gt_int[:,:,:,0]

    if out_float:
        newdict['float_center_x_'] = LOD_gt_float[:-1,:-1,:-1,0]
        newdict['float_center_y_'] = LOD_gt_float[:-1,:-1,:-1,1]
        newdict['float_center_z_'] = LOD_gt_float[:-1,:-1,:-1,2]

    if input_type=="sdf":
        newdict['input_sdf'] = LOD_input[:,:,:]
    elif input_type=="voxel":
        newdict['input_voxel'] = LOD_input[:-1,:-1,:-1]


    #augment data
    permutation_list = [ [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0] ]
    reversal_list = [ [0,0,0],[0,0,1],[0,1,0],[0,1,1], [1,0,0],[1,0,1],[1,1,0],[1,1,1] ]
    if aug_permutation:
        permutation = permutation_list[np.random.randint(len(permutation_list))]
    else:
        permutation = permutation_list[0]
    if aug_reversal:
        reversal = reversal_list[np.random.randint(len(reversal_list))]
    else:
        reversal = reversal_list[0]
    if aug_inversion:
        inversion_flag = np.random.randint(2)
    else:
        inversion_flag = 0


    if reversal[0]:
        for k in newdict: #inverse
            newdict[k] = newdict[k][::-1,:,:]
            if '_x_' in k:
                mask = (newdict[k]>=0)
                newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
    if reversal[1]:
        for k in newdict: #inverse
            newdict[k] = newdict[k][:,::-1,:]
            if '_y_' in k:
                mask = (newdict[k]>=0)
                newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
    if reversal[2]:
        for k in newdict: #inverse
            newdict[k] = newdict[k][:,:,::-1]
            if '_z_' in k:
                mask = (newdict[k]>=0)
                newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask


    if permutation == [0,1,2]:
        pass
    else:
        for k in newdict: #transpose
            newdict[k] = np.transpose(newdict[k], permutation)

        if out_float:
            olddict = newdict
            newdict = {}
            for k in olddict:
                newdict[k] = olddict[k]

            if permutation == [0,2,1]:
                newdict['float_center_y_'] = olddict['float_center_z_']
                newdict['float_center_z_'] = olddict['float_center_y_']

            elif permutation == [1,0,2]:
                newdict['float_center_x_'] = olddict['float_center_y_']
                newdict['float_center_y_'] = olddict['float_center_x_']

            elif permutation == [2,1,0]:
                newdict['float_center_x_'] = olddict['float_center_z_']
                newdict['float_center_z_'] = olddict['float_center_x_']

            elif permutation == [1,2,0]:
                newdict['float_center_x_'] = olddict['float_center_y_']
                newdict['float_center_y_'] = olddict['float_center_z_']
                newdict['float_center_z_'] = olddict['float_center_x_']

            elif permutation == [2,0,1]:
                newdict['float_center_x_'] = olddict['float_center_z_']
                newdict['float_center_y_'] = olddict['float_center_x_']
                newdict['float_center_z_'] = olddict['float_center_y_']


    #store outputs
    if out_bool:
        LOD_gt_int = np.zeros([grid_size_1,grid_size_1,grid_size_1,1], np.int32)
        if inversion_flag:
            LOD_gt_int[:,:,:,0] = 1-newdict['int_V_signs']
        else:
            LOD_gt_int[:,:,:,0] = newdict['int_V_signs']
    else:
        LOD_gt_int = None

    if out_float:
        LOD_gt_float = np.full([grid_size_1,grid_size_1,grid_size_1,3], -1, np.float32)
        LOD_gt_float[:-1,:-1,:-1,0] = newdict['float_center_x_']
        LOD_gt_float[:-1,:-1,:-1,1] = newdict['float_center_y_']
        LOD_gt_float[:-1,:-1,:-1,2] = newdict['float_center_z_']
    else:
        LOD_gt_float = None

    if input_type=="sdf":
        LOD_input = np.ones([grid_size_1,grid_size_1,grid_size_1], np.float32)
        if inversion_flag:
            LOD_input[:,:,:] = -newdict['input_sdf']
        else:
            LOD_input[:,:,:] = newdict['input_sdf']

    elif input_type=="voxel":
        LOD_input = np.zeros([grid_size_1,grid_size_1,grid_size_1], np.uint8)
        if inversion_flag:
            LOD_input[:-1,:-1,:-1] = 1-newdict['input_voxel']
        else:
            LOD_input[:-1,:-1,:-1] = newdict['input_voxel']

    return LOD_gt_int, LOD_gt_float, LOD_input



#this is not an efficient implementation. just for testing!
def dual_contouring_ndc_test(int_grid, float_grid):
    all_vertices = []
    all_triangles = []

    int_grid = np.squeeze(int_grid)
    dimx,dimy,dimz = int_grid.shape
    vertices_grid = np.full([dimx,dimy,dimz], -1, np.int32)

    #all vertices
    for i in range(0,dimx-1):
        for j in range(0,dimy-1):
            for k in range(0,dimz-1):
            
                v0 = int_grid[i,j,k]
                v1 = int_grid[i+1,j,k]
                v2 = int_grid[i+1,j+1,k]
                v3 = int_grid[i,j+1,k]
                v4 = int_grid[i,j,k+1]
                v5 = int_grid[i+1,j,k+1]
                v6 = int_grid[i+1,j+1,k+1]
                v7 = int_grid[i,j+1,k+1]
                
                color = v0
                if v1!=v0 or v2!=v0 or v3!=v0 or v4!=v0 or v5!=v0 or v6!=v0 or v7!=v0:
                    #add a vertex
                    vertices_grid[i,j,k] = len(all_vertices)
                    pos = float_grid[i,j,k]+np.array([i,j,k], np.float32)
                    all_vertices.append(pos)

    all_vertices = np.array(all_vertices, np.float32)


    #all triangles

    #i-direction
    for i in range(0,dimx-1):
        for j in range(1,dimy-1):
            for k in range(1,dimz-1):
                v0 = int_grid[i,j,k]
                v1 = int_grid[i+1,j,k]
                if v0!=v1:
                    if v0==0:
                        all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k],vertices_grid[i,j,k-1]])
                        all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j-1,k],vertices_grid[i,j,k]])
                    else:
                        all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k-1],vertices_grid[i,j,k]])
                        all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k],vertices_grid[i,j-1,k]])

    #j-direction
    for i in range(1,dimx-1):
        for j in range(0,dimy-1):
            for k in range(1,dimz-1):
                v0 = int_grid[i,j,k]
                v1 = int_grid[i,j+1,k]
                if v0!=v1:
                    if v0==0:
                        all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i,j,k-1],vertices_grid[i,j,k]])
                        all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i,j,k],vertices_grid[i-1,j,k]])
                    else:
                        all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i,j,k],vertices_grid[i,j,k-1]])
                        all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i-1,j,k],vertices_grid[i,j,k]])

    #k-direction
    for i in range(1,dimx-1):
        for j in range(1,dimy-1):
            for k in range(0,dimz-1):
                v0 = int_grid[i,j,k]
                v1 = int_grid[i,j,k+1]
                if v0!=v1:
                    if v0==0:
                        all_triangles.append([vertices_grid[i-1,j-1,k],vertices_grid[i-1,j,k],vertices_grid[i,j,k]])
                        all_triangles.append([vertices_grid[i-1,j-1,k],vertices_grid[i,j,k],vertices_grid[i,j-1,k]])
                    else:
                        all_triangles.append([vertices_grid[i-1,j-1,k],vertices_grid[i,j,k],vertices_grid[i-1,j,k]])
                        all_triangles.append([vertices_grid[i-1,j-1,k],vertices_grid[i,j-1,k],vertices_grid[i,j,k]])

    all_triangles = np.array(all_triangles, np.int32)

    return all_vertices, all_triangles



def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
    fout.close()

def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()

def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    fout.close()

def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
    fout.close()
