import numpy as np


#this is not an efficient implementation. just for testing!
def dual_contouring_47_test(int_grid, float_grid):
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


