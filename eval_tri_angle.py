import numpy as np
import os
import h5py
from multiprocessing import Process, Queue
import queue
import time
import trimesh
from sklearn.neighbors import KDTree


pred_dir = "samples/"
gt_dir = "../../objs/"

def load_obj(dire):
    fin = open(dire,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = []
    
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            vertices.append([x,y,z])
        if line[0] == 'f':
            if len(line)!=4:
                print("obj: len(line)!=4")
                exit(-1)
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)

    return vertices, triangles


def get_v_t_count(q, name_list):
    name_num = len(name_list)
    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        gt_obj_name = name_list[nid][2]
        pred_obj_name = name_list[nid][3]


        #load gt
        #load pred
        v,t = load_obj(pred_obj_name)
        
        slot = np.zeros([180],np.int32)

        for i in range(len(t)):
            for j in range(3):
                v1i = t[i,(j)%3]
                v2i = t[i,(j+1)%3]
                v3i = t[i,(j+2)%3]
                
                v1 = v[v1i]
                v2 = v[v2i]
                v3 = v[v3i]

                v12 = v2-v1
                v13 = v3-v1

                s_a = np.sqrt( np.sum(v12*v12)*np.sum(v13*v13) )
                if s_a<1e-10:
                    s_a = 1e-10
                cos_a = np.sum(v12*v13)/s_a
                if cos_a>1: cos_a=1
                if cos_a<-1: cos_a=-1
                a = (np.arccos(cos_a)/np.pi*180).astype(np.int32)
                if a>=180:
                    a = 180-1
                slot[a] += 1


        print(idx)
        q.put([idx,slot])


if __name__ == '__main__':

    fin = open("abc_obj_list.txt", 'r')
    obj_names = [name.strip() for name in fin.readlines()]
    obj_names = obj_names[int(len(obj_names)*0.8):]
    fin.close()

    obj_names_len = len(obj_names)

    slots_180 = np.zeros([180],np.int32)


    #prepare list of names
    num_of_process = 16
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
        workers.append(Process(target=get_v_t_count, args = (q, list_of_names)))

    for p in workers:
        p.start()


    counter = 0
    while True:
        item_flag = True
        try:
            idx, slot = q.get(True, 1.0)
        except queue.Empty:
            item_flag = False
        
        if item_flag:
            #process result
            slots_180 = slots_180 + slot
            counter += 1

        allExited = True
        for p in workers:
            if p.exitcode is None:
                allExited = False
                break
        if allExited and q.empty():
            break

    if counter!=obj_names_len:
        print("ERROR: counter!=obj_names_len")
        print(counter,obj_names_len)

    #accumulate, percentage
    acc = 0
    for idx in range(180):
        acc += slots_180[idx]
        slots_180[idx] = acc
    slots_180 = slots_180.astype(np.float32)/acc


    fout = open("result_tri_angle.txt", 'w')
    for idx in range(180):
        fout.write(str(slots_180[idx])+"\n")
    fout.close()

    print("finished")

