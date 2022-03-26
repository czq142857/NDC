#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

# Cython specific imports
import numpy as np
cimport numpy as np
import cython
np.import_array()


def dual_contouring_ndc(int[:,:,:,::1] int_grid, float[:,:,:,::1] float_grid):

    #arrays to store vertices and triangles
    #will grow dynamically according to the number of actual vertices and triangles
    cdef int all_vertices_len = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 16384
    cdef int all_triangles_max = 16384
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
    cdef float[:,::1] all_vertices = all_vertices_
    cdef int[:,::1] all_triangles = all_triangles_
    cdef float[:,::1] all_vertices_old = all_vertices_
    cdef int[:,::1] all_triangles_old = all_triangles_

    cdef int dimx,dimy,dimz
    dimx = int_grid.shape[0] -1
    dimy = int_grid.shape[1] -1
    dimz = int_grid.shape[2] -1

    #array for fast indexing vertices
    vertices_grid_ = np.full([dimx,dimy,dimz], -1, np.int32)
    cdef int[:,:,::1] vertices_grid = vertices_grid_

    cdef int i,j,k,ii,v0,v1,v2,v3,v4,v5,v6,v7


    #all vertices
    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i+1,j,k,0]
                v2 = int_grid[i+1,j+1,k,0]
                v3 = int_grid[i,j+1,k,0]
                v4 = int_grid[i,j,k+1,0]
                v5 = int_grid[i+1,j,k+1,0]
                v6 = int_grid[i+1,j+1,k+1,0]
                v7 = int_grid[i,j+1,k+1,0]

                if v1!=v0 or v2!=v0 or v3!=v0 or v4!=v0 or v5!=v0 or v6!=v0 or v7!=v0:
                    #add a vertex
                    vertices_grid[i,j,k] = all_vertices_len

                    #grow all_vertices
                    if all_vertices_len+1>=all_vertices_max:
                        all_vertices_max = all_vertices_max*2
                        all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
                        all_vertices = all_vertices_
                        for ii in range(all_vertices_len):
                            all_vertices[ii,0] = all_vertices_old[ii,0]
                            all_vertices[ii,1] = all_vertices_old[ii,1]
                            all_vertices[ii,2] = all_vertices_old[ii,2]
                        all_vertices_old = all_vertices_
                    
                    #add to all_vertices
                    all_vertices[all_vertices_len,0] = float_grid[i,j,k,0]+i
                    all_vertices[all_vertices_len,1] = float_grid[i,j,k,1]+j
                    all_vertices[all_vertices_len,2] = float_grid[i,j,k,2]+k
                    all_vertices_len += 1


    #all triangles

    #i-direction
    for i in range(0,dimx):
        for j in range(1,dimy):
            for k in range(1,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i+1,j,k,0]
                if v0!=v1:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    if v0==0:
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                    else:
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                        all_triangles_len += 1

    #j-direction
    for i in range(1,dimx):
        for j in range(0,dimy):
            for k in range(1,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i,j+1,k,0]
                if v0!=v1:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    if v0==0:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                        all_triangles_len += 1
                    else:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                        all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    #k-direction
    for i in range(1,dimx):
        for j in range(1,dimy):
            for k in range(0,dimz):
                v0 = int_grid[i,j,k,0]
                v1 = int_grid[i,j,k+1,0]
                if v0!=v1:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    if v0==0:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                        all_triangles_len += 1
                    else:
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                        all_triangles_len += 1
                        all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                        all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                        all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                        all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len]




def dual_contouring_undc(int[:,:,:,::1] int_grid, float[:,:,:,::1] float_grid):

    #arrays to store vertices and triangles
    #will grow dynamically according to the number of actual vertices and triangles
    cdef int all_vertices_len = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 16384
    cdef int all_triangles_max = 16384
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
    cdef float[:,::1] all_vertices = all_vertices_
    cdef int[:,::1] all_triangles = all_triangles_
    cdef float[:,::1] all_vertices_old = all_vertices_
    cdef int[:,::1] all_triangles_old = all_triangles_

    cdef int dimx,dimy,dimz
    dimx = int_grid.shape[0] -1
    dimy = int_grid.shape[1] -1
    dimz = int_grid.shape[2] -1

    #array for fast indexing vertices
    vertices_grid_ = np.full([dimx,dimy,dimz], -1, np.int32)
    cdef int[:,:,::1] vertices_grid = vertices_grid_

    cdef int i,j,k,ii


    #all vertices
    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                if int_grid[i,j,k,0] or int_grid[i,j+1,k,0] or int_grid[i,j+1,k+1,0] or int_grid[i,j,k+1,0] or int_grid[i,j,k,1] or int_grid[i+1,j,k,1] or int_grid[i+1,j,k+1,1] or int_grid[i,j,k+1,1] or int_grid[i,j,k,2] or int_grid[i+1,j,k,2] or int_grid[i+1,j+1,k,2] or int_grid[i,j+1,k,2]:
                    #add a vertex
                    vertices_grid[i,j,k] = all_vertices_len

                    #grow all_vertices
                    if all_vertices_len+1>=all_vertices_max:
                        all_vertices_max = all_vertices_max*2
                        all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
                        all_vertices = all_vertices_
                        for ii in range(all_vertices_len):
                            all_vertices[ii,0] = all_vertices_old[ii,0]
                            all_vertices[ii,1] = all_vertices_old[ii,1]
                            all_vertices[ii,2] = all_vertices_old[ii,2]
                        all_vertices_old = all_vertices_
                    
                    #add to all_vertices
                    all_vertices[all_vertices_len,0] = float_grid[i,j,k,0]+i
                    all_vertices[all_vertices_len,1] = float_grid[i,j,k,1]+j
                    all_vertices[all_vertices_len,2] = float_grid[i,j,k,2]+k
                    all_vertices_len += 1


    #all triangles

    #i-direction
    for i in range(0,dimx):
        for j in range(1,dimy):
            for k in range(1,dimz):
                if int_grid[i,j,k,0]:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                    all_triangles[all_triangles_len,1] = vertices_grid[i,j,k-1]
                    all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = vertices_grid[i,j-1,k-1]
                    all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                    all_triangles[all_triangles_len,2] = vertices_grid[i,j-1,k]
                    all_triangles_len += 1

    #j-direction
    for i in range(1,dimx):
        for j in range(0,dimy):
            for k in range(1,dimz):
                if int_grid[i,j,k,1]:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                    all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                    all_triangles[all_triangles_len,2] = vertices_grid[i,j,k-1]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = vertices_grid[i-1,j,k-1]
                    all_triangles[all_triangles_len,1] = vertices_grid[i-1,j,k]
                    all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                    all_triangles_len += 1

    #k-direction
    for i in range(1,dimx):
        for j in range(1,dimy):
            for k in range(0,dimz):
                if int_grid[i,j,k,2]:

                    #grow all_triangles
                    if all_triangles_len+2>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_

                    #add to all_triangles
                    all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                    all_triangles[all_triangles_len,1] = vertices_grid[i,j,k]
                    all_triangles[all_triangles_len,2] = vertices_grid[i-1,j,k]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = vertices_grid[i-1,j-1,k]
                    all_triangles[all_triangles_len,1] = vertices_grid[i,j-1,k]
                    all_triangles[all_triangles_len,2] = vertices_grid[i,j,k]
                    all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len]