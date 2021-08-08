import argparse

import os
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=400, type=int, help="Epoch to train [400]")
parser.add_argument("--lr", action="store", dest="lr", default=0.0001, type=float, help="Learning rate [0.0001]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="../groundtruth/gt_NDC", help="Root directory of dataset [gt_NDC]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the image samples [samples]")
parser.add_argument("--train_joint", action="store_true", dest="train_joint", default=False, help="Training both bool and float with one network [False]")
parser.add_argument("--train_bool", action="store_true", dest="train_bool", default=False, help="Training only bool with one network [False]")
parser.add_argument("--train_float", action="store_true", dest="train_float", default=False, help="Training only float with one network [False]")
parser.add_argument("--test_joint", action="store_true", dest="test_joint", default=False, help="Testing both bool and float with one network [False]")
parser.add_argument("--test_bool", action="store_true", dest="test_bool", default=False, help="Testing only bool with one network, using GT float [False]")
parser.add_argument("--test_float", action="store_true", dest="test_float", default=False, help="Testing only float with one network, using GT bool [False]")
parser.add_argument("--test_bool_float", action="store_true", dest="test_bool_float", default=False, help="Testing both bool and float with two networks [False]")
parser.add_argument("--input_type", action="store", dest="input_type", default="sdf", help="Input type, sdf or voxel [sdf]")
parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="to use which GPU [0]")

FLAGS = parser.parse_args()

is_training = False
is_testing = False
if FLAGS.train_joint or FLAGS.train_bool or FLAGS.train_float:
    is_training = True
if FLAGS.test_joint or FLAGS.test_bool or FLAGS.test_float or FLAGS.test_bool_float:
    is_testing = True
dataset_bool = False
dataset_float = False
net_bool = False
net_float = False
net_joint = False
if FLAGS.train_joint or FLAGS.test_joint:
    dataset_bool = True
    dataset_float = True
    net_joint = True
if FLAGS.train_bool or FLAGS.test_bool:
    dataset_bool = True
    net_bool = True
if FLAGS.train_float or FLAGS.test_float:
    dataset_float = True
    net_float = True
if FLAGS.test_bool_float:
    dataset_bool = True
    dataset_float = True
    net_bool = True
    net_float = True


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

import dataset
import model
import utils

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')



#Create network
if FLAGS.input_type == "sdf":
    CNN_3d = model.CNN_3d_rec7
elif FLAGS.input_type == "voxel":
    CNN_3d = model.CNN_3d_rec15

if net_joint:
    network_joint = CNN_3d(out_bool=True, out_float=True)
    network_joint.to(device)
if net_bool:
    network_bool = CNN_3d(out_bool=True, out_float=False)
    network_bool.to(device)
if net_float:
    network_float = CNN_3d(out_bool=False, out_float=True)
    network_float.to(device)


if is_training:
    #Create train/test dataset
    dataset_train = dataset.ABC_ndc_hdf5(FLAGS.data_dir, train=True, input_type=FLAGS.input_type, out_bool=dataset_bool, out_float=dataset_float)
    dataset_test = dataset.ABC_ndc_hdf5(FLAGS.data_dir, train=False, input_type=FLAGS.input_type, out_bool=True, out_float=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=16) #batch_size must be 1
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  #batch_size must be 1


    networks = []
    optimizers = []
    if net_joint:
        networks.append(network_joint)
        optimizers.append(torch.optim.Adam(network_joint.parameters()))
    if net_bool:
        networks.append(network_bool)
        optimizers.append(torch.optim.Adam(network_bool.parameters()))
    if net_float:
        networks.append(network_float)
        optimizers.append(torch.optim.Adam(network_float.parameters()))

    start_time = time.time()
    for epoch in range(FLAGS.epoch):
        for network in networks:
            network.train()

        if epoch%100==0:
            for optimizer in optimizers:
                for g in optimizer.param_groups:
                    lr = FLAGS.lr/(2**(epoch//100))
                    print("Setting learning rate to", lr)
                    g['lr'] = lr

        avg_loss = 0
        avg_acc_bool = 0
        avg_acc_float = 0
        avg_loss_count = 0
        avg_acc_bool_count = 0
        avg_acc_float_count = 0
        for i, data in enumerate(dataloader_train, 0):
            if dataset_bool and dataset_float:
                gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data
                gt_input = gt_input_.to(device)
                gt_output_bool = gt_output_bool_.to(device)
                gt_output_bool_mask = gt_output_bool_mask_.to(device)
                gt_output_float = gt_output_float_.to(device)
                gt_output_float_mask = gt_output_float_mask_.to(device)
            elif dataset_bool:
                gt_input_, gt_output_bool_, gt_output_bool_mask_ = data
                gt_input = gt_input_.to(device)
                gt_output_bool = gt_output_bool_.to(device)
                gt_output_bool_mask = gt_output_bool_mask_.to(device)
            elif dataset_float:
                gt_input_, gt_output_float_, gt_output_float_mask_ = data
                gt_input = gt_input_.to(device)
                gt_output_float = gt_output_float_.to(device)
                gt_output_float_mask = gt_output_float_mask_.to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            if net_joint:
                pred_output_bool, pred_output_float = network_joint(gt_input)
            if net_bool:
                pred_output_bool = network_bool(gt_input)
            if net_float:
                pred_output_float = network_float(gt_input)

            if dataset_bool:
                
                #binary cross encropy
                bool_mask_sum = torch.sum(gt_output_bool_mask)
                loss_bool = - torch.sum(( gt_output_bool*torch.log(torch.clamp(pred_output_bool, min=1e-10)) + (1-gt_output_bool)*torch.log(torch.clamp(1-pred_output_bool, min=1e-10)) )*gt_output_bool_mask)/torch.clamp(bool_mask_sum,min=1)
                acc_bool = torch.sum(( gt_output_bool*(pred_output_bool>0.5).float() + (1-gt_output_bool)*(pred_output_bool<=0.5).float() )*gt_output_bool_mask)/torch.clamp(bool_mask_sum,min=1)

            if dataset_float:
                
                #MSE
                loss_float = torch.sum(( (pred_output_float-gt_output_float)**2 )*gt_output_float_mask)/torch.clamp(torch.sum(gt_output_float_mask),min=1)

            if dataset_bool and dataset_float:
                if bool_mask_sum.data>0:
                    loss = loss_bool + loss_float
                    avg_acc_bool += acc_bool.data
                    avg_acc_bool_count += 1
                    avg_acc_float += loss_float.data
                    avg_acc_float_count += 1
                else:
                    loss = loss_float
                    avg_acc_float += loss_float.data
                    avg_acc_float_count += 1
            elif dataset_bool:
                loss = loss_bool
                avg_acc_bool += acc_bool.data
                avg_acc_bool_count += 1
            elif dataset_float:
                loss = loss_float
                avg_acc_float += loss_float.data
                avg_acc_float_count += 1
                
            avg_loss += loss.data
            avg_loss_count += 1

            loss.backward()
            for optimizer in optimizers:
                optimizer.step()


        print('[%d/%d] time: %.0f  loss: %.8f  loss_bool: %.8f  loss_float: %.8f' % (epoch, FLAGS.epoch, time.time()-start_time, avg_loss/max(avg_loss_count,1), avg_acc_bool/max(avg_acc_bool_count,1), avg_acc_float/max(avg_acc_float_count,1)))
        
        if epoch%50==49:
            #save weights
            print('saving net...')
            if net_joint:
                torch.save(network_joint.state_dict(), 'network_joint.pth')
            if net_bool:
                torch.save(network_bool.state_dict(), 'network_bool.pth')
            if net_float:
                torch.save(network_float.state_dict(), 'network_float.pth')
            print('saving net... complete')

            #test
            for network in networks:
                network.eval()
            counter = 0
            for i, data in enumerate(dataloader_test, 0):
                gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data
                gt_input = gt_input_.to(device)

                with torch.no_grad():
                    if net_joint:
                        pred_output_bool, pred_output_float = network_joint(gt_input)
                    if net_bool:
                        pred_output_bool = network_bool(gt_input)
                    if net_float:
                        pred_output_float = network_float(gt_input)

                batch_size = gt_input_.size()[0]
                for j in range(batch_size):
                    receptive_padding=3
                    if dataset_bool:
                        gt_input_numpy = gt_input_[j,0,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding].detach().cpu().numpy()
                        if FLAGS.input_type == "sdf":
                            pred_output_bool_numpy = np.expand_dims((gt_input_numpy<0).astype(np.int32), axis=3)
                        elif FLAGS.input_type == "voxel":
                            pred_output_bool_numpy = np.transpose(pred_output_bool[j].detach().cpu().numpy(), [1,2,3,0])
                            pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                            gt_output_bool_mask_numpy = gt_output_bool_mask_[j].detach().cpu().numpy()
                            pred_output_bool_numpy[:,:,:,0] = pred_output_bool_numpy[:,:,:,0]*gt_output_bool_mask_numpy[0] + gt_input_numpy*(1-gt_output_bool_mask_numpy[0])
                    else:
                        pred_output_bool_numpy = np.transpose(gt_output_bool_[j].detach().cpu().numpy(), [1,2,3,0])
                        pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                    if dataset_float:
                        pred_output_float_numpy = np.transpose(pred_output_float[j].detach().cpu().numpy(), [1,2,3,0])
                    else:
                        pred_output_float_numpy = np.transpose(gt_output_float_[j].detach().cpu().numpy(), [1,2,3,0])
                    pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
                    vertices, triangles = utils.dual_contouring_ndc_test(pred_output_bool_numpy, pred_output_float_numpy)
                    utils.write_obj_triangle(FLAGS.sample_dir+"/test_"+str(counter)+".obj", vertices, triangles)
                    counter += 1
                if counter>=32: break


elif is_testing:
    import cutils

    #Create test dataset
    if dataset_bool and dataset_float: #only read input
        dataset_test = dataset.ABC_ndc_hdf5(FLAGS.data_dir, train=False, input_type=FLAGS.input_type, out_bool=True, out_float=True, input_only=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  #batch_size must be 1
    else:
        dataset_test = dataset.ABC_ndc_hdf5(FLAGS.data_dir, train=False, input_type=FLAGS.input_type, out_bool=True, out_float=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=16)  #batch_size must be 1


    #load weights
    print('loading net...')
    if net_joint:
        network_joint.load_state_dict(torch.load('network_joint.pth'))
        print('network_joint loaded')
    if net_bool and FLAGS.input_type == "voxel":
        network_bool.load_state_dict(torch.load('network_bool.pth'))
        print('network_bool loaded')
    if net_float:
        network_float.load_state_dict(torch.load('network_float.pth'))
        print('network_float loaded')
    print('loading net... complete')

    #test
    if net_joint:
        network_joint.eval()
    if net_bool:
        network_bool.eval()
    if net_float:
        network_float.eval()

    counter = 0
    for i, data in enumerate(dataloader_test, 0):
        gt_input_, gt_output_bool_, gt_output_bool_mask_, gt_output_float_, gt_output_float_mask_ = data
        gt_input = gt_input_.to(device)

        with torch.no_grad():
            if net_joint:
                pred_output_bool, pred_output_float = network_joint(gt_input)
            if net_bool and FLAGS.input_type == "voxel":
                pred_output_bool = network_bool(gt_input)
            if net_float:
                pred_output_float = network_float(gt_input)

        batch_size = gt_input_.size()[0]
        for j in range(batch_size):
            print(counter)
            receptive_padding=3
            if dataset_bool:
                gt_input_numpy = gt_input_[j,0,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding,receptive_padding:-receptive_padding].detach().cpu().numpy()
                if FLAGS.input_type == "sdf":
                    pred_output_bool_numpy = np.expand_dims((gt_input_numpy<0).astype(np.int32), axis=3)
                elif FLAGS.input_type == "voxel":
                    pred_output_bool_numpy = np.transpose(pred_output_bool[j].detach().cpu().numpy(), [1,2,3,0])
                    pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
                    gt_output_bool_mask_numpy = gt_output_bool_mask_[j].detach().cpu().numpy()
                    pred_output_bool_numpy[:,:,:,0] = pred_output_bool_numpy[:,:,:,0]*gt_output_bool_mask_numpy[0] + gt_input_numpy*(1-gt_output_bool_mask_numpy[0])
            else:
                pred_output_bool_numpy = np.transpose(gt_output_bool_[j].detach().cpu().numpy(), [1,2,3,0])
                pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
            if dataset_float:
                pred_output_float_numpy = np.transpose(pred_output_float[j].detach().cpu().numpy(), [1,2,3,0])
            else:
                pred_output_float_numpy = np.transpose(gt_output_float_[j].detach().cpu().numpy(), [1,2,3,0])
            pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
            #vertices, triangles = utils.dual_contouring_ndc_test(pred_output_bool_numpy, pred_output_float_numpy)
            vertices, triangles = cutils.dual_contouring_ndc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
            utils.write_obj_triangle(FLAGS.sample_dir+"/test_"+str(counter)+".obj", vertices, triangles)
            counter += 1
        #if counter>=32: break




