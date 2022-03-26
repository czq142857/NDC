import numpy as np


#this is not an efficient implementation. just for testing!
def dual_contouring_undc_test(int_grid, float_grid):
    all_vertices = []
    all_triangles = []

    dimx,dimy,dimz,_ = int_grid.shape
    vertices_grid = np.full([dimx,dimy,dimz], -1, np.int32)

    #all vertices
    for i in range(0,dimx-1):
        for j in range(0,dimy-1):
            for k in range(0,dimz-1):
            
                ex0 = int_grid[i,j,k,0]
                ex1 = int_grid[i,j+1,k,0]
                ex2 = int_grid[i,j+1,k+1,0]
                ex3 = int_grid[i,j,k+1,0]

                ey0 = int_grid[i,j,k,1]
                ey1 = int_grid[i+1,j,k,1]
                ey2 = int_grid[i+1,j,k+1,1]
                ey3 = int_grid[i,j,k+1,1]

                ez0 = int_grid[i,j,k,2]
                ez1 = int_grid[i+1,j,k,2]
                ez2 = int_grid[i+1,j+1,k,2]
                ez3 = int_grid[i,j+1,k,2]
                
                if ex0 or ex1 or ex2 or ex3 or ey0 or ey1 or ey2 or ey3 or ez0 or ez1 or ez2 or ez3:
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
                if int_grid[i,j,k,0]:
                    all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k-1],vertices_grid[i,j,k]])
                    all_triangles.append([vertices_grid[i,j-1,k-1],vertices_grid[i,j,k],vertices_grid[i,j-1,k]])

    #j-direction
    for i in range(1,dimx-1):
        for j in range(0,dimy-1):
            for k in range(1,dimz-1):
                if int_grid[i,j,k,1]:
                    all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i,j,k],vertices_grid[i,j,k-1]])
                    all_triangles.append([vertices_grid[i-1,j,k-1],vertices_grid[i-1,j,k],vertices_grid[i,j,k]])

    #k-direction
    for i in range(1,dimx-1):
        for j in range(1,dimy-1):
            for k in range(0,dimz-1):
                if int_grid[i,j,k,2]:
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


def read_intersectionpn_file_as_2d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#intersectionpn'):
        raise IOError('Not an intersectionpn file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    point_nums = np.array(list(map(int, fp.readline().strip().split(b' '))),np.int32)
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape([np.sum(point_nums),6])
    fp.close()
    separated = []
    count = 0
    for i in range(len(point_nums)):
        separated.append(np.ascontiguousarray(data[count:count+point_nums[i]]))
        count += point_nums[i]
    return separated

#read sdf files produced by SDFGen
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

#a simplified reader for binvox files
#mostly copied from binvox_rw.py
#https://github.com/dimatura/binvox-rw-py
def read_binvox_file_as_3d_array(name,fix_coords=True):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        data = np.transpose(data, (0, 2, 1))
    data = np.ascontiguousarray(data, np.uint8)
    fp.close()
    return data


