#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

# Cython specific imports
import numpy as np
cimport numpy as np
import cython
np.import_array()



#note: treat each cube as an voxel, whose vertices are sampled SDF points
#e.g., 65 SDF grid -> 64 voxel grid
def get_input_voxel(char[:,:,::1] vox, int vox_size, int out_size, char[:,:,::1] out):
    cdef int dimx,dimy,dimz,upscale
    cdef int i,j,k,si,sj,sk,ei,ej,ek,x,y,z,v

    dimx = vox.shape[0]
    dimy = vox.shape[1]
    dimz = vox.shape[2]
    upscale = vox_size//out_size

    for i in range(out_size):
        for j in range(out_size):
            for k in range(out_size):
                si = i*upscale
                sj = j*upscale
                sk = k*upscale
                if si<0: si=0
                if sj<0: sj=0
                if sk<0: sk=0
                ei = (i+1)*upscale
                ej = (j+1)*upscale
                ek = (k+1)*upscale
                if ei>vox_size: ei=vox_size
                if ej>vox_size: ej=vox_size
                if ek>vox_size: ek=vox_size
                ei = ei+1
                ej = ej+1
                ek = ek+1

                v = 0
                for x in range(si,ei):
                    for y in range(sj,ej):
                        for z in range(sk,ek):
                            if vox[x,y,z]==1:
                                v=1
                                break
                        if v==1: break
                    if v==1: break
                out[i,j,k] = v



#put messy intersection points and their normals into buckets (cells), so that all points in a selected cell can be efficiently queried
def get_intersection_points_normals_in_cells(float[:,::1] inter_X, float[:,::1] inter_Y, float[:,::1] inter_Z, int grid_size_1, int[::1] inter_p, float[:,::1] inter_data):
    cdef int inter_X_size,inter_Y_size,inter_Z_size
    cdef int t,cx,cy,cz,idx
    cdef float px,py,pz

    cdef int grid_size_1_sqr = grid_size_1*grid_size_1
    cdef int current_len = grid_size_1*grid_size_1*grid_size_1

    inter_X_size = inter_X.shape[0]
    inter_Y_size = inter_Y.shape[0]
    inter_Z_size = inter_Z.shape[0]


    for t in range(inter_X_size):
        px = inter_X[t,0]
        py = inter_X[t,1]
        pz = inter_X[t,2]
        cx = (<int>px)
        cy = (<int>py)
        cz = (<int>pz)
        if cx>=0 and cx<grid_size_1 and cy>=0 and cy<grid_size_1 and cz>=0 and cz<grid_size_1:
            idx = cx*grid_size_1_sqr + cy*grid_size_1 + cz
            if inter_p[idx]>=0: inter_p[current_len] = inter_p[idx]
            inter_p[idx] = current_len
            inter_data[current_len,0] = px
            inter_data[current_len,1] = py
            inter_data[current_len,2] = pz
            inter_data[current_len,3] = inter_X[t,3]
            inter_data[current_len,4] = inter_X[t,4]
            inter_data[current_len,5] = inter_X[t,5]
            current_len += 1

    for t in range(inter_Y_size):
        px = inter_Y[t,0]
        py = inter_Y[t,1]
        pz = inter_Y[t,2]
        cx = (<int>px)
        cy = (<int>py)
        cz = (<int>pz)
        if cx>=0 and cx<grid_size_1 and cy>=0 and cy<grid_size_1 and cz>=0 and cz<grid_size_1:
            idx = cx*grid_size_1_sqr + cy*grid_size_1 + cz
            if inter_p[idx]>=0: inter_p[current_len] = inter_p[idx]
            inter_p[idx] = current_len
            inter_data[current_len,0] = px
            inter_data[current_len,1] = py
            inter_data[current_len,2] = pz
            inter_data[current_len,3] = inter_Y[t,3]
            inter_data[current_len,4] = inter_Y[t,4]
            inter_data[current_len,5] = inter_Y[t,5]
            current_len += 1

    for t in range(inter_Z_size):
        px = inter_Z[t,0]
        py = inter_Z[t,1]
        pz = inter_Z[t,2]
        cx = (<int>px)
        cy = (<int>py)
        cz = (<int>pz)
        if cx>=0 and cx<grid_size_1 and cy>=0 and cy<grid_size_1 and cz>=0 and cz<grid_size_1:
            idx = cx*grid_size_1_sqr + cy*grid_size_1 + cz
            if inter_p[idx]>=0: inter_p[current_len] = inter_p[idx]
            inter_p[idx] = current_len
            inter_data[current_len,0] = px
            inter_data[current_len,1] = py
            inter_data[current_len,2] = pz
            inter_data[current_len,3] = inter_Z[t,3]
            inter_data[current_len,4] = inter_Z[t,4]
            inter_data[current_len,5] = inter_Z[t,5]
            current_len += 1



def retrieve_intersection_points_normals_from_cells(int grid_size_1, int[::1] inter_p, float[:,::1] inter_data, int si, int sj, int sk, float[:,::1] reference_V):
    cdef int p_count, i, x,y,z
    cdef float ui,uj,uk, vi,vj,vk
    
    p_count = 0
    
    ui = <float>si
    uj = <float>sj
    uk = <float>sk
    vi = <float>(si+1)
    vj = <float>(sj+1)
    vk = <float>(sk+1)
    
    for x in range(2):
        for y in range(2):
            for z in range(2):
                i = (si+x)*grid_size_1*grid_size_1 + (sj+y)*grid_size_1 + (sk+z)
                while inter_p[i]>=0:
                    i = inter_p[i]
                    if inter_data[i,0]<=vi and inter_data[i,1]<=vj and inter_data[i,2]<=vk:
                        reference_V[p_count,0] = inter_data[i,0]-ui
                        reference_V[p_count,1] = inter_data[i,1]-uj
                        reference_V[p_count,2] = inter_data[i,2]-uk
                        reference_V[p_count,3] = inter_data[i,3]
                        reference_V[p_count,4] = inter_data[i,4]
                        reference_V[p_count,5] = inter_data[i,5]
                        p_count += 1
    
    return p_count



