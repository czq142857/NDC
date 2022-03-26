import os
import numpy as np
from multiprocessing import Process, Queue
import queue
import time



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
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            triangles.append([x-1,y-1,z-1])
    
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)
    
    return vertices, triangles

def write_obj(dire, vertices, triangles):
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()



def augment_objs(q, name_list):
    name_num = len(name_list)

    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        in_name = name_list[nid][2]


        print(pid,idx,in_name)
        v,t = load_obj(in_name+"/model.obj")
        
        for s in [10,9,8,7,6,5]:
            for i in [0,1]:
                for j in [0,1]:
                    for k in [0,1]:
                        newdir = in_name+"_"+str(s)+"_"+str(i)+"_"+str(j)+"_"+str(k)
                        os.makedirs(newdir)
                        vvv = v*(s/10.0)
                        if i==1:
                            vvv[:,0] = vvv[:,0]+0.5/64
                        if j==1:
                            vvv[:,1] = vvv[:,1]+0.5/64
                        if k==1:
                            vvv[:,2] = vvv[:,2]+0.5/64
                        write_obj(newdir+"/model.obj",vvv,t)

        q.put([1,pid,idx])




if __name__ == '__main__':

    target_dir = "../objs/"
    if not os.path.exists(target_dir):
        print("ERROR: this dir does not exist: "+target_dir)
        exit()

    obj_names_x = os.listdir(target_dir)

    fin = open("abc_obj_list.txt", 'r')
    obj_names = [name.strip() for name in fin.readlines()]
    fin.close()

    for name in obj_names_x:
        if name not in obj_names:
            os.system("rm -r "+target_dir+name)

    #augment training set only
    obj_names = obj_names[:int(len(obj_names)*0.8)]
    obj_names_len = len(obj_names)

    #prepare list of names
    even_distribution = [16]
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
        in_name = target_dir + obj_names[idx]

        list_of_list_of_names[process_id].append([process_id, idx, in_name])

    
    #map processes
    q = Queue()
    workers = []
    for i in range(P_start,P_end):
        list_of_names = list_of_list_of_names[i]
        workers.append(Process(target=augment_objs, args = (q, list_of_names)))

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
    


