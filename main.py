import argparse
import os
import numpy as np
import time

import torch


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", action="store", dest="epoch", default=400, type=int, help="Epoch to train [400,250,25]")
parser.add_argument("--lr", action="store", dest="lr", default=0.0001, type=float, help="Learning rate [0.0001]")
parser.add_argument("--lr_half_life", action="store", dest="lr_half_life", default=100, type=int, help="Halve lr every few epochs [100,5]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./groundtruth/gt_NDC", help="Root directory of dataset [gt_NDC,gt_UNDC,gt_UNDCa]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="weights", help="Directory name to save the checkpoints [weights]")
parser.add_argument("--checkpoint_save_frequency", action="store", dest="checkpoint_save_frequency", default=50, type=int, help="Save checkpoint every few epochs [50,10,1]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the output samples [samples]")

parser.add_argument("--train_bool", action="store_true", dest="train_bool", default=False, help="Training only bool with one network [False]")
parser.add_argument("--train_float", action="store_true", dest="train_float", default=False, help="Training only float with one network [False]")

parser.add_argument("--test_bool", action="store_true", dest="test_bool", default=False, help="Testing only bool with one network, using GT float [False]")
parser.add_argument("--test_float", action="store_true", dest="test_float", default=False, help="Testing only float with one network, using GT bool [False]")
parser.add_argument("--test_bool_float", action="store_true", dest="test_bool_float", default=False, help="Testing both bool and float with two networks [False]")
parser.add_argument("--test_input", action="store", dest="test_input", default="", help="Select an input file for quick testing [*.sdf, *.binvox, *.ply, *.hdf5]")

parser.add_argument("--point_num", action="store", dest="point_num", default=4096, type=int, help="Size of input point cloud for testing [4096,16384,524288]")
parser.add_argument("--grid_size", action="store", dest="grid_size", default=64, type=int, help="Size of output grid for testing [32,64,128]")
parser.add_argument("--block_num_per_dim", action="store", dest="block_num_per_dim", default=5, type=int, help="Number of blocks per dimension [1,5,10]")
parser.add_argument("--block_padding", action="store", dest="block_padding", default=5, type=int, help="Padding for each block [5]")

parser.add_argument("--input_type", action="store", dest="input_type", default="sdf", help="Input type [sdf,voxel,udf,pointcloud,noisypc]")
parser.add_argument("--method", action="store", dest="method", default="ndc", help="Method type [ndc,undc,ndcx]")
parser.add_argument("--postprocessing", action="store_true", dest="postprocessing", default=False, help="Enable the post-processing step to close small holes [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")

FLAGS = parser.parse_args()

is_training = False #training on a dataset
is_testing = False #testing on a dataset
quick_testing = False #testing on a single shape/scene
if FLAGS.train_bool or FLAGS.train_float:
    is_training = True
if FLAGS.test_bool or FLAGS.test_float or FLAGS.test_bool_float:
    is_testing = True

net_bool = False
net_float = False
if FLAGS.train_bool or FLAGS.test_bool:
    net_bool = True
if FLAGS.train_float or FLAGS.test_float:
    net_float = True
if FLAGS.test_bool_float:
    net_bool = True
    net_float = True

if FLAGS.test_input != "":
    quick_testing = True
    net_bool = True
    net_float = True


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

import dataset
import datasetpc
import model
import modelpc
import utils

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')



#Create network
if FLAGS.method == "ndc":
    if FLAGS.input_type == "sdf":
        CNN_3d = model.CNN_3d_rec7
    elif FLAGS.input_type == "voxel":
        CNN_3d = model.CNN_3d_rec15
elif FLAGS.method == "undc":
    if FLAGS.input_type == "sdf":
        CNN_3d = model.CNN_3d_rec7
    elif FLAGS.input_type == "voxel":
        CNN_3d = model.CNN_3d_rec15
    elif FLAGS.input_type == "udf":
        CNN_3d = model.CNN_3d_rec7
    elif FLAGS.input_type == "pointcloud":
        CNN_3d = modelpc.local_pointnet
    elif FLAGS.input_type == "noisypc":
        CNN_3d = modelpc.local_pointnet_larger
elif FLAGS.method == "ndcx":
    if FLAGS.input_type == "sdf":
        CNN_3d = model.CNN_3d_rec7_resnet
    elif FLAGS.input_type == "voxel":
        CNN_3d = model.CNN_3d_rec15_resnet

#Create network
receptive_padding = 3 #for grid input
pooling_radius = 2 #for pointcloud input
KNN_num = modelpc.KNN_num

if net_bool:
    if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":
        network_bool = CNN_3d(out_bool=True, out_float=False, is_undc=(FLAGS.method == "undc"))
    elif FLAGS.input_type == "pointcloud" or FLAGS.input_type == "noisypc":
        network_bool = CNN_3d(out_bool=True, out_float=False)
    network_bool.to(device)
if net_float:
    network_float = CNN_3d(out_bool=False, out_float=True)
    network_float.to(device)


def worker_init_fn(worker_id):
    np.random.seed(int(time.time()*10000000)%10000000 + worker_id)


if is_training:
    if net_bool and net_float:
        print("ERROR: net_bool and net_float cannot both be activated in training")
        exit(-1)

    #Create train/test dataset
    if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":
        dataset_train = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, FLAGS.input_type, train=True, out_bool=net_bool, out_float=net_float, is_undc=(FLAGS.method == "undc"))
        dataset_test = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, FLAGS.input_type, train=False, out_bool=True, out_float=True, is_undc=(FLAGS.method == "undc"))
    elif FLAGS.input_type == "pointcloud" or FLAGS.input_type == "noisypc":
        dataset_train = datasetpc.ABC_pointcloud_hdf5(FLAGS.data_dir, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, FLAGS.input_type, train=True, out_bool=net_bool, out_float=net_float)
        dataset_test = datasetpc.ABC_pointcloud_hdf5(FLAGS.data_dir, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, FLAGS.input_type, train=False, out_bool=True, out_float=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=16, worker_init_fn=worker_init_fn) #batch_size must be 1
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  #batch_size must be 1


    if net_bool:
        optimizer = torch.optim.Adam(network_bool.parameters())
    if net_float:
        optimizer = torch.optim.Adam(network_float.parameters())

    start_time = time.time()
    for epoch in range(FLAGS.epoch):
        if net_bool:
            network_bool.train()
        if net_float:
            network_float.train()

        if epoch%FLAGS.lr_half_life==0:
            for g in optimizer.param_groups:
                lr = FLAGS.lr/(2**(epoch//FLAGS.lr_half_life))
                print("Setting learning rate to", lr)
                g['lr'] = lr

        avg_loss = 0
        avg_acc_bool = 0
        avg_acc_float = 0
        avg_loss_count = 0
        avg_acc_bool_count = 0
        avg_acc_float_count = 0
        for i, data in enumerate(dataloader_train, 0):

            if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":

                if net_bool:
                    gt_input_, gt_output_bool_, gt_output_bool_mask_ = data

                    gt_input = gt_input_.to(device)
                    gt_output_bool = gt_output_bool_.to(device)
                    gt_output_bool_mask = gt_output_bool_mask_.to(device)

                    optimizer.zero_grad()

                    pred_output_bool = network_bool(gt_input)

                    #binary cross encropy
                    bool_mask_sum = torch.sum(gt_output_bool_mask)
                    loss_bool = - torch.sum(( gt_output_bool*torch.log(torch.clamp(pred_output_bool, min=1e-10)) + (1-gt_output_bool)*torch.log(torch.clamp(1-pred_output_bool, min=1e-10)) )*gt_output_bool_mask)/torch.clamp(bool_mask_sum,min=1)
                    acc_bool = torch.sum(( gt_output_bool*(pred_output_bool>0.5).float() + (1-gt_output_bool)*(pred_output_bool<=0.5).float() )*gt_output_bool_mask)/torch.clamp(bool_mask_sum,min=1)
                    
                    loss = loss_bool
                    avg_acc_bool += acc_bool.item()
                    avg_acc_bool_count += 1

                if net_float:
                    gt_input_, gt_output_float_, gt_output_float_mask_ = data

                    gt_input = gt_input_.to(device)
                    gt_output_float = gt_output_float_.to(device)
                    gt_output_float_mask = gt_output_float_mask_.to(device)

                    optimizer.zero_grad()

                    pred_output_float = network_float(gt_input)

                    #MSE
                    loss_float = torch.sum(( (pred_output_float-gt_output_float)**2 )*gt_output_float_mask)/torch.clamp(torch.sum(gt_output_float_mask),min=1)

                    loss = loss_float
                    avg_acc_float += loss_float.item()
                    avg_acc_float_count += 1

            elif FLAGS.input_type == "pointcloud" or FLAGS.input_type == "noisypc":

                if net_bool:
                    pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_, gt_output_bool_ = data
                    
                    pc_KNN_idx = pc_KNN_idx_[0].to(device)
                    pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
                    voxel_xyz_int = voxel_xyz_int_[0].to(device)
                    voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
                    voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)
                    gt_output_bool = gt_output_bool_[0].to(device)

                    optimizer.zero_grad()

                    pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                    #binary cross encropy
                    loss_bool = - torch.mean( gt_output_bool*torch.log(torch.clamp(pred_output_bool, min=1e-10)) + (1-gt_output_bool)*torch.log(torch.clamp(1-pred_output_bool, min=1e-10)) )
                    acc_bool = torch.mean( gt_output_bool*(pred_output_bool>0.5).float() + (1-gt_output_bool)*(pred_output_bool<=0.5).float() )

                    loss = loss_bool
                    avg_acc_bool += acc_bool.item()
                    avg_acc_bool_count += 1

                elif net_float:
                    pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_, gt_output_float_,gt_output_float_mask_ = data
                    
                    pc_KNN_idx = pc_KNN_idx_[0].to(device)
                    pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
                    voxel_xyz_int = voxel_xyz_int_[0].to(device)
                    voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
                    voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)
                    gt_output_float = gt_output_float_[0].to(device)
                    gt_output_float_mask = gt_output_float_mask_[0].to(device)

                    optimizer.zero_grad()

                    pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                    #MSE
                    loss_float = torch.sum(( (pred_output_float-gt_output_float)**2 )*gt_output_float_mask )/torch.clamp(torch.sum(gt_output_float_mask),min=1)

                    loss = loss_float
                    avg_acc_float += loss_float.item()
                    avg_acc_float_count += 1


            avg_loss += loss.item()
            avg_loss_count += 1

            loss.backward()
            optimizer.step()


        print('[%d/%d] time: %.0f  loss: %.8f  loss_bool: %.8f  loss_float: %.8f' % (epoch, FLAGS.epoch, time.time()-start_time, avg_loss/max(avg_loss_count,1), avg_acc_bool/max(avg_acc_bool_count,1), avg_acc_float/max(avg_acc_float_count,1)))


        if epoch%FLAGS.checkpoint_save_frequency==FLAGS.checkpoint_save_frequency-1:
            #save weights
            print('saving net...')
            if net_bool:
                torch.save(network_bool.state_dict(), FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_bool.pth")
            if net_float:
                torch.save(network_float.state_dict(), FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth")
            print('saving net... complete')

            #test
            if net_bool:
                network_bool.eval()
            if net_float:
                network_float.eval()

            for i, data in enumerate(dataloader_test, 0):

                if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":

                    gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data
                    gt_input = gt_input_.to(device)

                    with torch.no_grad():
                        if net_bool:
                            pred_output_bool = network_bool(gt_input)
                        if net_float:
                            pred_output_float = network_float(gt_input)

                    if net_bool:
                        if FLAGS.method == "undc":
                            pred_output_bool_numpy = np.transpose(pred_output_bool[0].detach().cpu().numpy(), [1,2,3,0])
                            pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                            gt_output_bool_mask_numpy = np.transpose(gt_output_bool_mask_[0].detach().cpu().numpy(), [1,2,3,0]).astype(np.int32)
                            pred_output_bool_numpy = pred_output_bool_numpy*gt_output_bool_mask_numpy
                        else:
                            gt_input_numpy = gt_input_[0,0,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding].detach().cpu().numpy()
                            if FLAGS.input_type == "voxel":
                                pred_output_bool_numpy = np.transpose(pred_output_bool[0].detach().cpu().numpy(), [1,2,3,0])
                                pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                                gt_output_bool_mask_numpy = np.transpose(gt_output_bool_mask_[0].detach().cpu().numpy(), [1,2,3,0]).astype(np.int32)
                                gt_input_numpy = np.expand_dims(gt_input_numpy.astype(np.int32), axis=3)
                                pred_output_bool_numpy = pred_output_bool_numpy*gt_output_bool_mask_numpy + gt_input_numpy*(1-gt_output_bool_mask_numpy)
                            if FLAGS.input_type == "sdf":
                                pred_output_bool_numpy = np.expand_dims((gt_input_numpy<0).astype(np.int32), axis=3)
                    else:
                        pred_output_bool_numpy = np.transpose(gt_output_bool_[0].detach().cpu().numpy(), [1,2,3,0])
                        pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                    if net_float:
                        pred_output_float_numpy = np.transpose(pred_output_float[0].detach().cpu().numpy(), [1,2,3,0])
                    else:
                        pred_output_float_numpy = np.transpose(gt_output_float_[0].detach().cpu().numpy(), [1,2,3,0])

                elif FLAGS.input_type == "pointcloud" or FLAGS.input_type == "noisypc":

                    pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_, gt_output_bool_,gt_output_float_,_ = data

                    pred_output_bool_numpy = np.zeros([FLAGS.grid_size+1,FLAGS.grid_size+1,FLAGS.grid_size+1,3], np.float32)
                    pred_output_float_numpy = np.full([FLAGS.grid_size+1,FLAGS.grid_size+1,FLAGS.grid_size+1,3], 0.5, np.float32)

                    pc_KNN_idx = pc_KNN_idx_[0].to(device)
                    pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
                    voxel_xyz_int = voxel_xyz_int_[0].to(device)
                    voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
                    voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)

                    with torch.no_grad():
                        if net_bool:
                            pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
                        if net_float:
                            pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                    if net_bool:
                        pred_output_bool_ = pred_output_bool.detach().cpu().numpy()
                    else:
                        pred_output_bool_ = gt_output_bool_[0].numpy()
                    if net_float:
                        pred_output_float_ = pred_output_float.detach().cpu().numpy()
                    else:
                        pred_output_float_ = gt_output_float_[0].numpy()

                    voxel_xyz_int = voxel_xyz_int_[0].numpy()
                    pred_output_bool_numpy[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_output_bool_
                    pred_output_float_numpy[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_output_float_

                    pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)


                pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
                if FLAGS.method == "undc":
                    vertices, triangles = utils.dual_contouring_undc_test(pred_output_bool_numpy, pred_output_float_numpy)
                else:
                    vertices, triangles = utils.dual_contouring_ndc_test(pred_output_bool_numpy, pred_output_float_numpy)
                utils.write_obj_triangle(FLAGS.sample_dir+"/test_"+str(i)+".obj", vertices, triangles)

                if i>=32: break


elif is_testing:
    import cutils

    #Create test dataset
    if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":
        if net_bool and net_float: #only read input
            dataset_test = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, FLAGS.input_type, train=False, out_bool=True, out_float=True, is_undc=(FLAGS.method == "undc"), input_only=True)
        else:
            dataset_test = dataset.ABC_grid_hdf5(FLAGS.data_dir, FLAGS.grid_size, receptive_padding, FLAGS.input_type, train=False, out_bool=True, out_float=True, is_undc=(FLAGS.method == "undc"))
    elif FLAGS.input_type == "pointcloud" or FLAGS.input_type == "noisypc":
        if net_bool and net_float: #only read input
            dataset_test = datasetpc.ABC_pointcloud_hdf5(FLAGS.data_dir, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, FLAGS.input_type, train=False, out_bool=True, out_float=True, input_only=True)
        else:
            dataset_test = datasetpc.ABC_pointcloud_hdf5(FLAGS.data_dir, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, FLAGS.input_type, train=False, out_bool=True, out_float=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  #batch_size must be 1


    #load weights
    print('loading net...')
    if net_bool and (FLAGS.method == "undc" or FLAGS.input_type != "sdf"):
        network_bool.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_bool.pth"))
        print('network_bool weights loaded')
    if net_float:
        network_float.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth"))
        print('network_float weights loaded')
    print('loading net... complete')

    #test
    if net_bool:
        network_bool.eval()
    if net_float:
        network_float.eval()


    for i, data in enumerate(dataloader_test, 0):

        if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":

            gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data
            
            gt_input = gt_input_.to(device)
            if FLAGS.method == "undc":
                gt_output_bool_mask = gt_output_bool_mask_.to(device)

            with torch.no_grad():
                if net_bool:
                    pred_output_bool = network_bool(gt_input)
                if net_float:
                    pred_output_float = network_float(gt_input)

                if net_bool:

                    if FLAGS.method == "undc":
                        if FLAGS.input_type == "sdf" or FLAGS.input_type == "udf":
                            pred_output_bool = ( pred_output_bool[0] > 0.5 ).int()
                        if FLAGS.input_type == "voxel":
                            pred_output_bool = ( pred_output_bool[0]*gt_output_bool_mask[0] > 0.5 ).int()
                        pred_output_bool = pred_output_bool.permute(1,2,3,0)
                        if FLAGS.postprocessing:
                            pred_output_bool = modelpc.postprocessing(pred_output_bool)
                        pred_output_bool_numpy = pred_output_bool.detach().cpu().numpy().astype(np.int32)

                    else:
                        gt_input_numpy = gt_input_[0,0,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding].detach().cpu().numpy()
                        if FLAGS.input_type == "voxel":
                            pred_output_bool_numpy = np.transpose(pred_output_bool[0].detach().cpu().numpy(), [1,2,3,0])
                            pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                            gt_output_bool_mask_numpy = np.transpose(gt_output_bool_mask_[0].detach().cpu().numpy(), [1,2,3,0]).astype(np.int32)
                            gt_input_numpy = np.expand_dims(gt_input_numpy.astype(np.int32), axis=3)
                            pred_output_bool_numpy = pred_output_bool_numpy*gt_output_bool_mask_numpy + gt_input_numpy*(1-gt_output_bool_mask_numpy)
                        if FLAGS.input_type == "sdf":
                            pred_output_bool_numpy = np.expand_dims((gt_input_numpy<0).astype(np.int32), axis=3)

                else:
                    pred_output_bool_numpy = np.transpose(gt_output_bool_[0].detach().cpu().numpy(), [1,2,3,0])
                    pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)

                if net_float:
                    pred_output_float_numpy = np.transpose(pred_output_float[0].detach().cpu().numpy(), [1,2,3,0])
                else:
                    pred_output_float_numpy = np.transpose(gt_output_float_[0].detach().cpu().numpy(), [1,2,3,0])

        elif FLAGS.input_type == "pointcloud" or FLAGS.input_type == "noisypc":

            pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_, gt_output_bool_,gt_output_float_,_ = data

            pc_KNN_idx = pc_KNN_idx_[0].to(device)
            pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
            voxel_xyz_int = voxel_xyz_int_[0].to(device)
            voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
            voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)

            with torch.no_grad():
                if net_bool:
                    pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
                if net_float:
                    pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                if not net_bool:
                    pred_output_bool = gt_output_bool_[0].to(device)
                if not net_float:
                    pred_output_float = gt_output_float_[0].to(device)

                pred_output_bool_grid = torch.zeros([FLAGS.grid_size+1,FLAGS.grid_size+1,FLAGS.grid_size+1,3], dtype=torch.int32, device=device)
                pred_output_float_grid = torch.full([FLAGS.grid_size+1,FLAGS.grid_size+1,FLAGS.grid_size+1,3], 0.5, device=device)

                pred_output_bool_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = (pred_output_bool>0.5).int()
                pred_output_float_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_output_float

                if FLAGS.postprocessing:
                    pred_output_bool_grid = modelpc.postprocessing(pred_output_bool_grid)

                pred_output_bool_numpy = pred_output_bool_grid.detach().cpu().numpy()
                pred_output_float_numpy = pred_output_float_grid.detach().cpu().numpy()


        pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
        if FLAGS.method == "undc":
            #vertices, triangles = utils.dual_contouring_undc_test(pred_output_bool_numpy, pred_output_float_numpy)
            vertices, triangles = cutils.dual_contouring_undc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
        else:
            #vertices, triangles = utils.dual_contouring_ndc_test(pred_output_bool_numpy, pred_output_float_numpy)
            vertices, triangles = cutils.dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
        utils.write_obj_triangle(FLAGS.sample_dir+"/test_"+str(i)+".obj", vertices, triangles)

        #if i>=32: break


elif quick_testing:
    import cutils


    #load weights
    print('loading net...')
    if net_bool and (FLAGS.method == "undc" or FLAGS.input_type != "sdf"):
        network_bool.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_bool.pth"))
        print('network_bool weights loaded')
    if net_float:
        network_float.load_state_dict(torch.load(FLAGS.checkpoint_dir+"/weights_"+FLAGS.method+"_"+FLAGS.input_type+"_float.pth"))
        print('network_float weights loaded')
    print('loading net... complete')

    #test
    if net_bool:
        network_bool.eval()
    if net_float:
        network_float.eval()


    if FLAGS.input_type == "sdf" or FLAGS.input_type == "voxel" or FLAGS.input_type == "udf":
        #Create test dataset
        dataset_test = dataset.single_shape_grid(FLAGS.test_input, receptive_padding, FLAGS.input_type, is_undc=(FLAGS.method == "undc"))
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)  #batch_size must be 1

        for i, data in enumerate(dataloader_test, 0):

            gt_input_, gt_output_bool_mask_ = data

            gt_input = gt_input_.to(device)
            if FLAGS.method == "undc":
                gt_output_bool_mask = gt_output_bool_mask_.to(device)

            with torch.no_grad():
                if net_bool:
                    pred_output_bool = network_bool(gt_input)
                if net_float:
                    pred_output_float = network_float(gt_input)

                if net_bool:

                    if FLAGS.method == "undc":
                        if FLAGS.input_type == "sdf" or FLAGS.input_type == "udf":
                            pred_output_bool = ( pred_output_bool[0] > 0.5 ).int()
                        if FLAGS.input_type == "voxel":
                            pred_output_bool = ( pred_output_bool[0]*gt_output_bool_mask[0] > 0.5 ).int()
                        pred_output_bool = pred_output_bool.permute(1,2,3,0)
                        if FLAGS.postprocessing:
                            pred_output_bool = modelpc.postprocessing(pred_output_bool)
                        pred_output_bool_numpy = pred_output_bool.detach().cpu().numpy().astype(np.int32)

                    else:
                        gt_input_numpy = gt_input_[0,0,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding].detach().cpu().numpy()
                        if FLAGS.input_type == "voxel":
                            pred_output_bool_numpy = np.transpose(pred_output_bool[0].detach().cpu().numpy(), [1,2,3,0])
                            pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                            gt_output_bool_mask_numpy = np.transpose(gt_output_bool_mask_[0].detach().cpu().numpy(), [1,2,3,0]).astype(np.int32)
                            gt_input_numpy = np.expand_dims(gt_input_numpy.astype(np.int32), axis=3)
                            pred_output_bool_numpy = pred_output_bool_numpy*gt_output_bool_mask_numpy + gt_input_numpy*(1-gt_output_bool_mask_numpy)
                        if FLAGS.input_type == "sdf":
                            pred_output_bool_numpy = np.expand_dims((gt_input_numpy<0).astype(np.int32), axis=3)

                else:
                    pred_output_bool_numpy = np.transpose(gt_output_bool_[0].detach().cpu().numpy(), [1,2,3,0])
                    pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)

                if net_float:
                    pred_output_float_numpy = np.transpose(pred_output_float[0].detach().cpu().numpy(), [1,2,3,0])
                else:
                    pred_output_float_numpy = np.transpose(gt_output_float_[0].detach().cpu().numpy(), [1,2,3,0])


    elif FLAGS.input_type == "pointcloud":
        #Create test dataset
        dataset_test = datasetpc.single_shape_pointcloud(FLAGS.test_input, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, normalize=False)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)  #batch_size must be 1

        for i, data in enumerate(dataloader_test, 0):

            pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_ = data

            pc_KNN_idx = pc_KNN_idx_[0].to(device)
            pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
            voxel_xyz_int = voxel_xyz_int_[0].to(device)
            voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
            voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)

            with torch.no_grad():
                if net_bool:
                    pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
                if net_float:
                    pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                if not net_bool:
                    pred_output_bool = gt_output_bool_[0].to(device)
                if not net_float:
                    pred_output_float = gt_output_float_[0].to(device)

                pred_output_bool_grid = torch.zeros([FLAGS.grid_size+1,FLAGS.grid_size+1,FLAGS.grid_size+1,3], dtype=torch.int32, device=device)
                pred_output_float_grid = torch.full([FLAGS.grid_size+1,FLAGS.grid_size+1,FLAGS.grid_size+1,3], 0.5, device=device)

                pred_output_bool_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = (pred_output_bool>0.5).int()
                pred_output_float_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_output_float

                if FLAGS.postprocessing:
                    pred_output_bool_grid = modelpc.postprocessing(pred_output_bool_grid)

                pred_output_bool_numpy = pred_output_bool_grid.detach().cpu().numpy()
                pred_output_float_numpy = pred_output_float_grid.detach().cpu().numpy()


    elif FLAGS.input_type == "noisypc":
        #Create test dataset
        dataset_test = datasetpc.scene_crop_pointcloud(FLAGS.test_input, FLAGS.point_num, FLAGS.grid_size, KNN_num, pooling_radius, FLAGS.block_num_per_dim, FLAGS.block_padding)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)  #batch_size must be 1

        #create large grid
        full_scene_size = np.copy(dataset_test.full_scene_size)
        pred_output_bool_numpy = np.zeros([FLAGS.grid_size*full_scene_size[0],FLAGS.grid_size*full_scene_size[1],FLAGS.grid_size*full_scene_size[2],3], np.int32)
        pred_output_float_numpy = np.zeros([FLAGS.grid_size*full_scene_size[0],FLAGS.grid_size*full_scene_size[1],FLAGS.grid_size*full_scene_size[2],3], np.float32)

        full_size = full_scene_size[0]*full_scene_size[1]*full_scene_size[2]
        for i, data in enumerate(dataloader_test, 0):
            print(i,"/",full_size)
            pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_ = data

            if pc_KNN_idx_.size()[1]==1: continue

            idx_x = i//(full_scene_size[1]*full_scene_size[2])
            idx_yz = i%(full_scene_size[1]*full_scene_size[2])
            idx_y = idx_yz//full_scene_size[2]
            idx_z = idx_yz%full_scene_size[2]

            pc_KNN_idx = pc_KNN_idx_[0].to(device)
            pc_KNN_xyz = pc_KNN_xyz_[0].to(device)
            voxel_xyz_int = voxel_xyz_int_[0].to(device)
            voxel_KNN_idx = voxel_KNN_idx_[0].to(device)
            voxel_KNN_xyz = voxel_KNN_xyz_[0].to(device)

            with torch.no_grad():
                if net_bool:
                    pred_output_bool = network_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
                if net_float:
                    pred_output_float = network_float(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)

                if not net_bool:
                    pred_output_bool = gt_output_bool_[0].to(device)
                if not net_float:
                    pred_output_float = gt_output_float_[0].to(device)

                pred_output_bool_grid = torch.zeros([FLAGS.grid_size*2+1,FLAGS.grid_size*2+1,FLAGS.grid_size*2+1,3], dtype=torch.int32, device=device)
                pred_output_float_grid = torch.full([FLAGS.grid_size*2+1,FLAGS.grid_size*2+1,FLAGS.grid_size*2+1,3], 0.5, device=device)

                pred_output_bool_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = (pred_output_bool>0.3).int()
                pred_output_float_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_output_float

                if FLAGS.postprocessing:
                    pred_output_bool_grid = modelpc.postprocessing(pred_output_bool_grid)

                pred_output_bool_numpy[idx_x*FLAGS.grid_size:(idx_x+1)*FLAGS.grid_size, idx_y*FLAGS.grid_size:(idx_y+1)*FLAGS.grid_size, idx_z*FLAGS.grid_size:(idx_z+1)*FLAGS.grid_size] = pred_output_bool_grid[FLAGS.block_padding:FLAGS.block_padding+FLAGS.grid_size,FLAGS.block_padding:FLAGS.block_padding+FLAGS.grid_size,FLAGS.block_padding:FLAGS.block_padding+FLAGS.grid_size].detach().cpu().numpy()
                pred_output_float_numpy[idx_x*FLAGS.grid_size:(idx_x+1)*FLAGS.grid_size, idx_y*FLAGS.grid_size:(idx_y+1)*FLAGS.grid_size, idx_z*FLAGS.grid_size:(idx_z+1)*FLAGS.grid_size] = pred_output_float_grid[FLAGS.block_padding:FLAGS.block_padding+FLAGS.grid_size,FLAGS.block_padding:FLAGS.block_padding+FLAGS.grid_size,FLAGS.block_padding:FLAGS.block_padding+FLAGS.grid_size].detach().cpu().numpy()


    pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
    if FLAGS.method == "undc":
        #vertices, triangles = utils.dual_contouring_undc_test(pred_output_bool_numpy, pred_output_float_numpy)
        vertices, triangles = cutils.dual_contouring_undc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
    else:
        #vertices, triangles = utils.dual_contouring_ndc_test(pred_output_bool_numpy, pred_output_float_numpy)
        vertices, triangles = cutils.dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
    utils.write_obj_triangle(FLAGS.sample_dir+"/quicktest_"+FLAGS.method+"_"+FLAGS.input_type+".obj", vertices, triangles)
