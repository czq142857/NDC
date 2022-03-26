import numpy as np
import os
import h5py
from multiprocessing import Process, Queue
import queue
import time
import trimesh
from sklearn.neighbors import KDTree


sample_num = 100000
pred_dir = "samples/"
gt_dir = "../../objs/"
f1_threshold = 0.003
ef1_radius = 0.004
ef1_dotproduct_threshold = 0.2
ef1_threshold = 0.005


def get_cd_nc_f1_ecd_ef1(q, name_list):
    name_num = len(name_list)
    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        gt_obj_name = name_list[nid][2]
        pred_obj_name = name_list[nid][3]


        #load gt
        gt_mesh = trimesh.load(gt_obj_name)
        gt_points, gt_indexs = gt_mesh.sample(sample_num, return_index=True)
        gt_normals = gt_mesh.face_normals[gt_indexs]
        #load pred
        pred_mesh = trimesh.load(pred_obj_name)
        pred_points, pred_indexs = pred_mesh.sample(sample_num, return_index=True)
        pred_points = pred_points/64-0.5
        pred_normals = pred_mesh.face_normals[pred_indexs]

        #cd and nc and f1

        # from gt to pred
        pred_tree = KDTree(pred_points)
        dist, inds = pred_tree.query(gt_points, k=1)
        recall = np.sum(dist < f1_threshold) / float(len(dist))
        gt2pred_mean_cd1 = np.mean(dist)
        dist = np.square(dist)
        gt2pred_mean_cd2 = np.mean(dist)
        neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
        dotproduct = np.abs(np.sum(gt_normals*neighbor_normals, axis=1))
        gt2pred_nc = np.mean(dotproduct)

        gt2pred_na = []
        for i in range(90):
            gt2pred_na.append( np.mean( (dotproduct<np.cos(i/180.0*np.pi)).astype(np.float32) ) )

        # from pred to gt
        gt_tree = KDTree(gt_points)
        dist, inds = gt_tree.query(pred_points, k=1)
        precision = np.sum(dist < f1_threshold) / float(len(dist))
        pred2gt_mean_cd1 = np.mean(dist)
        dist = np.square(dist)
        pred2gt_mean_cd2 = np.mean(dist)
        neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
        dotproduct = np.abs(np.sum(pred_normals*neighbor_normals, axis=1))
        pred2gt_nc = np.mean(dotproduct)

        pred2gt_na = []
        for i in range(90):
            pred2gt_na.append( np.mean( (dotproduct<np.cos(i/180.0*np.pi)).astype(np.float32) ) )

        cd1 = gt2pred_mean_cd1+pred2gt_mean_cd1
        cd2 = gt2pred_mean_cd2+pred2gt_mean_cd2
        nc = (gt2pred_nc+pred2gt_nc)/2
        if recall+precision > 0: f1 = 2 * recall * precision / (recall + precision)
        else: f1 = 0


        #sample gt edge points
        indslist = gt_tree.query_radius(gt_points, ef1_radius)
        flags = np.zeros([len(gt_points)],np.bool)
        for p in range(len(gt_points)):
            inds = indslist[p]
            if len(inds)>0:
                this_normals = gt_normals[p:p+1]
                neighbor_normals = gt_normals[inds]
                dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
                if np.any(dotproduct < ef1_dotproduct_threshold):
                    flags[p] = True
        gt_edge_points = np.ascontiguousarray(gt_points[flags])

        #sample pred edge points
        indslist = pred_tree.query_radius(pred_points, ef1_radius)
        flags = np.zeros([len(pred_points)],np.bool)
        for p in range(len(pred_points)):
            inds = indslist[p]
            if len(inds)>0:
                this_normals = pred_normals[p:p+1]
                neighbor_normals = pred_normals[inds]
                dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
                if np.any(dotproduct < ef1_dotproduct_threshold):
                    flags[p] = True
        pred_edge_points = np.ascontiguousarray(pred_points[flags])

        #write_ply_point("temp/"+str(idx)+"_gt.ply", gt_edge_points)
        #write_ply_point("temp/"+str(idx)+"_pred.ply", pred_edge_points)

        #ecd ef1

        if len(pred_edge_points)==0: pred_edge_points=np.zeros([486,3],np.float32)
        if len(gt_edge_points)==0:
            ecd = 0
            ef1 = 1
        else:
            # from gt to pred
            tree = KDTree(pred_edge_points)
            dist, inds = tree.query(gt_edge_points, k=1)
            erecall = np.sum(dist < ef1_threshold) / float(len(dist))
            gt2pred_mean_ecd1 = np.mean(dist)
            dist = np.square(dist)
            gt2pred_mean_ecd2 = np.mean(dist)

            # from pred to gt
            tree = KDTree(gt_edge_points)
            dist, inds = tree.query(pred_edge_points, k=1)
            eprecision = np.sum(dist < ef1_threshold) / float(len(dist))
            pred2gt_mean_ecd1 = np.mean(dist)
            dist = np.square(dist)
            pred2gt_mean_ecd2 = np.mean(dist)

            ecd1 = gt2pred_mean_ecd1+pred2gt_mean_ecd1
            ecd2 = gt2pred_mean_ecd2+pred2gt_mean_ecd2
            if erecall+eprecision > 0: ef1 = 2 * erecall * eprecision / (erecall + eprecision)
            else: ef1 = 0

        print(idx,cd1,cd2,nc,f1,ecd1,ecd2,ef1)
        q.put([idx,gt2pred_mean_cd1,gt2pred_mean_cd2,pred2gt_mean_cd1,pred2gt_mean_cd2,cd1,cd2,gt2pred_nc,gt2pred_na,pred2gt_nc,pred2gt_na,nc,recall,precision,f1,gt2pred_mean_ecd1,gt2pred_mean_ecd2,pred2gt_mean_ecd1,pred2gt_mean_ecd2,ecd1,ecd2,erecall,eprecision,ef1])


if __name__ == '__main__':

    fin = open("abc_obj_list.txt", 'r')
    obj_names = [name.strip() for name in fin.readlines()]
    obj_names = obj_names[int(len(obj_names)*0.8):]
    fin.close()

    obj_names_len = len(obj_names)

    numbers_gt2pred_mean_cd1 = np.zeros([obj_names_len],np.float32)
    numbers_gt2pred_mean_cd2 = np.zeros([obj_names_len],np.float32)
    numbers_pred2gt_mean_cd1 = np.zeros([obj_names_len],np.float32)
    numbers_pred2gt_mean_cd2 = np.zeros([obj_names_len],np.float32)
    numbers_cd1 = np.zeros([obj_names_len],np.float32)
    numbers_cd2 = np.zeros([obj_names_len],np.float32)

    numbers_gt2pred_nc = np.zeros([obj_names_len],np.float32)
    numbers_gt2pred_na = np.zeros([obj_names_len,90],np.float32)
    numbers_pred2gt_nc = np.zeros([obj_names_len],np.float32)
    numbers_pred2gt_na = np.zeros([obj_names_len,90],np.float32)
    numbers_nc = np.zeros([obj_names_len],np.float32)

    numbers_recall = np.zeros([obj_names_len],np.float32)
    numbers_precision = np.zeros([obj_names_len],np.float32)
    numbers_f1 = np.zeros([obj_names_len],np.float32)

    numbers_gt2pred_mean_ecd1 = np.zeros([obj_names_len],np.float32)
    numbers_gt2pred_mean_ecd2 = np.zeros([obj_names_len],np.float32)
    numbers_pred2gt_mean_ecd1 = np.zeros([obj_names_len],np.float32)
    numbers_pred2gt_mean_ecd2 = np.zeros([obj_names_len],np.float32)
    numbers_ecd1 = np.zeros([obj_names_len],np.float32)
    numbers_ecd2 = np.zeros([obj_names_len],np.float32)

    numbers_erecall = np.zeros([obj_names_len],np.float32)
    numbers_eprecision = np.zeros([obj_names_len],np.float32)
    numbers_ef1 = np.zeros([obj_names_len],np.float32)

    start_time = time.time()

    #prepare list of names
    num_of_process = 20
    list_of_list_of_names = []
    for i in range(num_of_process):
        list_of_list_of_names.append([])
    for idx in range(obj_names_len):
        process_id = idx%num_of_process
        gt_obj_name = gt_dir + obj_names[idx] + "/model.obj"
        pred_obj_name = pred_dir + "test_" + str(idx) + ".obj"
        list_of_list_of_names[process_id].append([process_id, idx, gt_obj_name, pred_obj_name])

    
    #map processes
    q = Queue()
    workers = []
    for i in range(num_of_process):
        list_of_names = list_of_list_of_names[i]
        workers.append(Process(target=get_cd_nc_f1_ecd_ef1, args = (q, list_of_names)))

    for p in workers:
        p.start()


    counter = 0
    while True:
        item_flag = True
        try:
            idx,gt2pred_mean_cd1,gt2pred_mean_cd2,pred2gt_mean_cd1,pred2gt_mean_cd2,cd1,cd2,gt2pred_nc,gt2pred_na,pred2gt_nc,pred2gt_na,nc,recall,precision,f1,gt2pred_mean_ecd1,gt2pred_mean_ecd2,pred2gt_mean_ecd1,pred2gt_mean_ecd2,ecd1,ecd2,erecall,eprecision,ef1 = q.get(True, 1.0)
        except queue.Empty:
            item_flag = False
        
        if item_flag:
            #process result
            counter += 1
            numbers_gt2pred_mean_cd1[idx] = gt2pred_mean_cd1
            numbers_gt2pred_mean_cd2[idx] = gt2pred_mean_cd2
            numbers_pred2gt_mean_cd1[idx] = pred2gt_mean_cd1
            numbers_pred2gt_mean_cd2[idx] = pred2gt_mean_cd2
            numbers_cd1[idx] = cd1
            numbers_cd2[idx] = cd2

            numbers_gt2pred_nc[idx] = gt2pred_nc
            numbers_gt2pred_na[idx] = np.array(gt2pred_na)
            numbers_pred2gt_nc[idx] = pred2gt_nc
            numbers_pred2gt_na[idx] = np.array(pred2gt_na)
            numbers_nc[idx] = nc

            numbers_recall[idx] = recall
            numbers_precision[idx] = precision
            numbers_f1[idx] = f1

            numbers_gt2pred_mean_ecd1[idx] = gt2pred_mean_ecd1
            numbers_gt2pred_mean_ecd2[idx] = gt2pred_mean_ecd2
            numbers_pred2gt_mean_ecd1[idx] = pred2gt_mean_ecd1
            numbers_pred2gt_mean_ecd2[idx] = pred2gt_mean_ecd2
            numbers_ecd1[idx] = ecd1
            numbers_ecd2[idx] = ecd2

            numbers_erecall[idx] = erecall
            numbers_eprecision[idx] = eprecision
            numbers_ef1[idx] = ef1

        allExited = True
        for p in workers:
            if p.exitcode is None:
                allExited = False
                break
        if allExited and q.empty():
            break

    if counter!=obj_names_len:
        print("ERROR: counter!=obj_names_len")
        exit(-1)

    fout = open("result_100000.txt", 'w')

    fout.write(
        str(np.mean(numbers_gt2pred_mean_cd1))+"\t"
        +str(np.mean(numbers_gt2pred_mean_cd2))+"\t"
        +str(np.mean(numbers_pred2gt_mean_cd1))+"\t"
        +str(np.mean(numbers_pred2gt_mean_cd2))+"\t"
        +str(np.mean(numbers_cd1))+"\t"
        +str(np.mean(numbers_cd2))+"\t"
        +str(np.mean(numbers_gt2pred_nc))+"\t"
        +str(np.mean(numbers_pred2gt_nc))+"\t"
        +str(np.mean(numbers_nc))+"\t"
        +str(np.mean(numbers_recall))+"\t"
        +str(np.mean(numbers_precision))+"\t"
        +str(np.mean(numbers_f1))+"\t"
        +str(np.mean(numbers_gt2pred_mean_ecd1))+"\t"
        +str(np.mean(numbers_gt2pred_mean_ecd2))+"\t"
        +str(np.mean(numbers_pred2gt_mean_ecd1))+"\t"
        +str(np.mean(numbers_pred2gt_mean_ecd2))+"\t"
        +str(np.mean(numbers_ecd1))+"\t"
        +str(np.mean(numbers_ecd2))+"\t"
        +str(np.mean(numbers_erecall))+"\t"
        +str(np.mean(numbers_eprecision))+"\t"
        +str(np.mean(numbers_ef1))+"\n"
        )

    numbers_gt2pred_na = np.mean(numbers_gt2pred_na,0)
    numbers_pred2gt_na = np.mean(numbers_pred2gt_na,0)
    for i in range(90):
        fout.write(str(numbers_gt2pred_na[i])+"\t"+str(numbers_pred2gt_na[i])+"\n")

    print("finished, time:", time.time()-start_time)

