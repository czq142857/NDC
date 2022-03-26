import numpy as np
#import cv2
#import mcubes
import os
import h5py
from multiprocessing import Process, Queue
import queue
import time
import argparse
import trimesh

from scipy.linalg import lstsq
#Compute least-squares solution to equation Ax = b.
#Compute a vector x such that the 2-norm |b - A x| is minimized.
#Input a(M, N), b(M,)
#Output x(N,), ...

import utils
import cutils


#this script outputs LOD of ground truth
# grid_size     voxel_size  intersection_size
# 64            512         64
# 32            256         32

# each grid cell has 3 + 3 = 6 values, without repetition
# 3 cube edge flags (int)
# 1 cube internal P (float)



def get_gt_from_intersectionpn(q, name_list):
    name_num = len(name_list)

    cell_voxel_size = 8

    num_of_int_params = 3
    num_of_float_params = 3

    point_sample_num = 65536

    grid_size_list = [32,64]
    LOD_gt_int = {}
    LOD_gt_float = {}
    LOD_input_sdf = {}
    LOD_input_voxel = {}
    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        LOD_gt_int[grid_size] = np.zeros([grid_size_1,grid_size_1,grid_size_1,num_of_int_params], np.uint8)
        LOD_gt_float[grid_size] = np.zeros([grid_size_1,grid_size_1,grid_size_1,num_of_float_params], np.float32)
        LOD_input_sdf[grid_size] = np.ones([grid_size_1,grid_size_1,grid_size_1], np.float32)
        LOD_input_voxel[grid_size] = np.zeros([grid_size_1,grid_size_1,grid_size_1], np.uint8)


    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        in_name = name_list[nid][2]
        out_name = name_list[nid][3]

        in_obj_name = in_name + ".obj"
        in_sdf_name = in_name + ".sdf"
        in_intersection_name = in_name + ".intersectionpn"
        in_binvox_name = in_name + ".binvox"
        out_hdf5_name = out_name + ".hdf5"

        #if nid+1<name_num and os.path.exists(name_list[nid+1][3] + ".hdf5"): continue
        #if os.path.exists(out_hdf5_name): continue

        print(pid,'  ',nid,'/',name_num, idx, in_binvox_name)


        #run exe to get sdf, intersection, and binvox
        command = "./VOXGen "+in_obj_name+" 1024 0"
        os.system(command)
        command = "./SDFGen "+in_obj_name+" 128 0"
        os.system(command)
        command = "./IntersectionXYZpn "+in_obj_name+" 128 0"
        os.system(command)


        #read
        gt_mesh = trimesh.load(in_obj_name)
        gt_points = gt_mesh.sample(point_sample_num, return_index=False)
        np.random.shuffle(gt_points)

        sdf_129 = utils.read_sdf_file_as_3d_array(in_sdf_name) #128

        inter_X, inter_Y, inter_Z = utils.read_intersectionpn_file_as_2d_array(in_intersection_name) #128

        voxel_1025 = utils.read_binvox_file_as_3d_array(in_binvox_name,fix_coords=False) #1024

        #compute gt
        for grid_size in grid_size_list:
            grid_size_1 = grid_size+1
            voxel_size = grid_size*cell_voxel_size
            downscale = 1024//voxel_size

            start_time = time.time()

            #prepare downsampled voxels and intersections
            LOD_inter_X = np.copy(inter_X[ (inter_X[:,1]%downscale==0) & (inter_X[:,2]%downscale==0) ])
            LOD_inter_Y = np.copy(inter_Y[ (inter_Y[:,0]%downscale==0) & (inter_Y[:,2]%downscale==0) ])
            LOD_inter_Z = np.copy(inter_Z[ (inter_Z[:,0]%downscale==0) & (inter_Z[:,1]%downscale==0) ])
            LOD_inter_X[:,:3] = LOD_inter_X[:,:3]/downscale
            LOD_inter_Y[:,:3] = LOD_inter_Y[:,:3]/downscale
            LOD_inter_Z[:,:3] = LOD_inter_Z[:,:3]/downscale
            
            LOD_voxels = np.zeros([voxel_size+cell_voxel_size+1,voxel_size+cell_voxel_size+1,voxel_size+cell_voxel_size+1], np.uint8)
            LOD_voxels[0:voxel_size+1,0:voxel_size+1,0:voxel_size+1] = voxel_1025[0::downscale,0::downscale,0::downscale]


            tmp_sdf = sdf_129[0::downscale,0::downscale,0::downscale]
            LOD_input_sdf[grid_size][:] = tmp_sdf

            tmp_voxel = np.zeros([grid_size_1,grid_size_1,grid_size_1], np.uint8)
            cutils.get_input_voxel(voxel_1025, 1024, grid_size, tmp_voxel)
            LOD_input_voxel[grid_size][:] = tmp_voxel

            #prepare an efficient data structure to store intersections
            LOD_intersection_maxlen = len(LOD_inter_X)+len(LOD_inter_Y)+len(LOD_inter_Z) + (grid_size_1**3)*2
            LOD_intersection_pointer = np.full([LOD_intersection_maxlen], -1, np.int32)
            LOD_intersection_data = np.full([LOD_intersection_maxlen,6], -1, np.float32)
            cutils.get_intersection_points_normals_in_cells(LOD_inter_X, LOD_inter_Y, LOD_inter_Z, grid_size_1, LOD_intersection_pointer, LOD_intersection_data)

            #prepare arrays to store ground truth
            LOD_gt_tmp_int = np.full([grid_size_1,grid_size_1,grid_size_1,num_of_int_params], 0, np.uint8)
            LOD_gt_tmp_float = np.full([grid_size_1,grid_size_1,grid_size_1,num_of_float_params], -10, np.float32)


            # ----- get_gt -----
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                            reference_V = np.zeros([256,6], np.float32)
                            reference_V_len = cutils.retrieve_intersection_points_normals_from_cells(grid_size_1, LOD_intersection_pointer, LOD_intersection_data, i,j,k, reference_V)
                            if reference_V_len==0: #empty
                                continue
                            
                            if np.any( (reference_V[:reference_V_len,1]==0) & (reference_V[:reference_V_len,2]==0) ):
                                LOD_gt_tmp_int[i,j,k,0] = 1
                            if np.any( (reference_V[:reference_V_len,0]==0) & (reference_V[:reference_V_len,2]==0) ):
                                LOD_gt_tmp_int[i,j,k,1] = 1
                            if np.any( (reference_V[:reference_V_len,0]==0) & (reference_V[:reference_V_len,1]==0) ):
                                LOD_gt_tmp_int[i,j,k,2] = 1
                            
                            #prepare a(?,3) b(?)
                            #n(x-p)=0 --> nx=np
                            a = reference_V[:reference_V_len+3,3:6]
                            b = np.sum(reference_V[:reference_V_len+3,3:6] * reference_V[:reference_V_len+3,0:3], axis=1)
                            #add regularization to avoid singular
                            reg_scale = 1e-6
                            reg_pos = np.mean(reference_V[:reference_V_len,0:3],axis=0)
                            a[reference_V_len+0,:] = [reg_scale,0,0]
                            a[reference_V_len+1,:] = [0,reg_scale,0]
                            a[reference_V_len+2,:] = [0,0,reg_scale]
                            b[reference_V_len:reference_V_len+3] = reg_pos*reg_scale
                            x = lstsq(a,b)[0]

                            if np.min(x)<0 or np.max(x)>1:
                                tmp_scale = 1
                                while np.min(x)<0 or np.max(x)>1:
                                    if reg_scale>1000: break
                                    reg_scale = reg_scale+tmp_scale
                                    a[reference_V_len+0,:] = [reg_scale,0,0]
                                    a[reference_V_len+1,:] = [0,reg_scale,0]
                                    a[reference_V_len+2,:] = [0,0,reg_scale]
                                    b[reference_V_len:reference_V_len+3] = reg_pos*reg_scale
                                    x = lstsq(a,b)[0]

                                if reg_scale>1000:
                                    reg_scale = 1e-6
                                    reg_pos = np.array([0.5,0.5,0.5], np.float32)
                                    a[reference_V_len+0,:] = [reg_scale,0,0]
                                    a[reference_V_len+1,:] = [0,reg_scale,0]
                                    a[reference_V_len+2,:] = [0,0,reg_scale]
                                    b[reference_V_len:reference_V_len+3] = reg_pos*reg_scale
                                    x = lstsq(a,b)[0]
                                    if np.min(x)<0 or np.max(x)>1:
                                        tmp_scale = 1
                                        while np.min(x)<0 or np.max(x)>1:
                                            reg_scale = reg_scale+tmp_scale
                                            a[reference_V_len+0,:] = [reg_scale,0,0]
                                            a[reference_V_len+1,:] = [0,reg_scale,0]
                                            a[reference_V_len+2,:] = [0,0,reg_scale]
                                            b[reference_V_len:reference_V_len+3] = reg_pos*reg_scale
                                            x = lstsq(a,b)[0]

                                for c in range(10):
                                    tmp_scale = tmp_scale/2
                                    reg_scale = reg_scale-tmp_scale
                                    a[reference_V_len+0,:] = [reg_scale,0,0]
                                    a[reference_V_len+1,:] = [0,reg_scale,0]
                                    a[reference_V_len+2,:] = [0,0,reg_scale]
                                    b[reference_V_len:reference_V_len+3] = reg_pos*reg_scale
                                    x = lstsq(a,b)[0]
                                    if np.min(x)<0 or np.max(x)>1:
                                        reg_scale = reg_scale+tmp_scale

                                if np.min(x)<0 or np.max(x)>1:
                                    a[reference_V_len+0,:] = [reg_scale,0,0]
                                    a[reference_V_len+1,:] = [0,reg_scale,0]
                                    a[reference_V_len+2,:] = [0,0,reg_scale]
                                    b[reference_V_len:reference_V_len+3] = reg_pos*reg_scale
                                    x = lstsq(a,b)[0]


                            LOD_gt_tmp_float[i,j,k] = x
            # ----- get_gt end -----


            LOD_gt_int[grid_size][:] = LOD_gt_tmp_int
            LOD_gt_float[grid_size][:] = LOD_gt_tmp_float


            #print(time.time() - start_time)

            #vertices, triangles = utils.dual_contouring_undc_test(LOD_gt_int[grid_size][:,:,:,:], LOD_gt_float[grid_size][:,:,:,:])
            #vertices = vertices/grid_size-0.5
            #utils.write_obj_triangle(out_name+"_"+str(grid_size)+".obj", vertices, triangles)
            #utils.write_ply_point(out_name+"_"+str(grid_size)+".ply", gt_points)

            #os.system("cp "+in_obj_name+" "+out_name+"_x.obj")
            #tmp_pc = np.concatenate([LOD_inter_X,LOD_inter_Y,LOD_inter_Z], axis=0)
            #utils.write_ply_point_normal(out_name+"_"+str(grid_size)+".ply", tmp_pc)

        
        #record data
        hdf5_file = h5py.File(out_hdf5_name, 'w')
        hdf5_file.create_dataset("pointcloud", [point_sample_num,3], np.float32, compression=9)
        hdf5_file["pointcloud"][:] = gt_points
        for grid_size in grid_size_list:
            grid_size_1 = grid_size+1
            hdf5_file.create_dataset(str(grid_size)+"_int", [grid_size_1,grid_size_1,grid_size_1,num_of_int_params], np.uint8, compression=9)
            hdf5_file.create_dataset(str(grid_size)+"_float", [grid_size_1,grid_size_1,grid_size_1,num_of_float_params], np.float32, compression=9)
            hdf5_file.create_dataset(str(grid_size)+"_sdf", [grid_size_1,grid_size_1,grid_size_1], np.float32, compression=9)
            hdf5_file.create_dataset(str(grid_size)+"_voxel", [grid_size_1,grid_size_1,grid_size_1], np.uint8, compression=9)
            hdf5_file[str(grid_size)+"_int"][:] = LOD_gt_int[grid_size]
            hdf5_file[str(grid_size)+"_float"][:] = LOD_gt_float[grid_size]
            hdf5_file[str(grid_size)+"_sdf"][:] = LOD_input_sdf[grid_size]
            hdf5_file[str(grid_size)+"_voxel"][:] = LOD_input_voxel[grid_size]
        hdf5_file.close()
        

        #delete sdf, intersection, and binvox to save space
        os.remove(in_sdf_name)
        os.remove(in_intersection_name)
        os.remove(in_binvox_name)

        del voxel_1025


        q.put([1,pid,idx])




if __name__ == '__main__':

    target_dir = "../objs/"
    if not os.path.exists(target_dir):
        print("ERROR: this dir does not exist: "+target_dir)
        exit()

    write_dir = "./gt_UNDC/"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    #obj_names = os.listdir(target_dir)
    #obj_names = sorted(obj_names)

    #obj_names = ["00000016"]

    fin = open("abc_obj_list.txt", 'r')
    obj_names = [name.strip() for name in fin.readlines()]
    fin.close()

    obj_names_len = len(obj_names)


    #prepare list of names
    even_distribution = [20]
    this_machine_id = 0
    num_of_process = 0
    P_start = 0
    P_end = 0
    for i in range(len(even_distribution)):
        num_of_process += even_distribution[i]
        if i<this_machine_id:
            P_start += even_distribution[i]
        if i<=this_machine_id:
            P_end += even_distribution[i]
    print(this_machine_id, P_start, P_end)

    list_of_list_of_names = []
    for i in range(num_of_process):
        list_of_list_of_names.append([])
    for idx in range(obj_names_len):
        process_id = idx%num_of_process
        in_name = target_dir + obj_names[idx] + "/model"
        out_name = write_dir + obj_names[idx]

        list_of_list_of_names[process_id].append([process_id, idx, in_name, out_name])

    
    #map processes
    q = Queue()
    workers = []
    for i in range(P_start,P_end):
        list_of_names = list_of_list_of_names[i]
        workers.append(Process(target=get_gt_from_intersectionpn, args = (q, list_of_names)))

    for p in workers:
        p.start()


    counter = 0
    while True:
        item_flag = True
        try:
            success_flag,pid,idx = q.get(True, 1.0)
        except queue.Empty:
            item_flag = False
        
        if item_flag:
            #process result
            counter += success_flag

        allExited = True
        for p in workers:
            if p.exitcode is None:
                allExited = False
                break
        if allExited and q.empty():
            break


    print("finished")
    print("returned", counter,"/",obj_names_len)
    

    #q = Queue()
    #get_gt_from_intersectionpn(q,list_of_list_of_names[0])


