import os
import numpy as np
import time
import h5py

import torch

from utils import read_data,read_and_augment_data_ndc,read_and_augment_data_undc,read_data_input_only, read_sdf_file_as_3d_array,read_binvox_file_as_3d_array

class ABC_grid_hdf5(torch.utils.data.Dataset):
    def __init__(self, data_dir, output_grid_size, receptive_padding, input_type, train, out_bool, out_float, is_undc, input_only=False):
        self.data_dir = data_dir
        self.output_grid_size = output_grid_size
        self.receptive_padding = receptive_padding
        self.train = train
        self.input_type = input_type
        self.out_bool = out_bool
        self.out_float = out_float
        self.is_undc = is_undc
        self.input_only = input_only

        if self.out_bool and self.out_float and self.train:
            print("ERROR: out_bool and out_float cannot both be activated in training")
            exit(-1)

        #self.hdf5_names = os.listdir(self.data_dir)
        #self.hdf5_names = [name[:-5] for name in self.hdf5_names if name[-5:]==".hdf5"]
        #self.hdf5_names = sorted(self.hdf5_names)

        fin = open("abc_obj_list.txt", 'r')
        self.hdf5_names = [name.strip() for name in fin.readlines()]
        fin.close()

        if self.train:
            self.hdf5_names = self.hdf5_names[:int(len(self.hdf5_names)*0.8)]
            print("Total#", "train", len(self.hdf5_names), self.input_type, self.out_bool, self.out_float)
        else:
            self.hdf5_names = self.hdf5_names[int(len(self.hdf5_names)*0.8):]
            print("Total#", "test", len(self.hdf5_names), self.input_type, self.out_bool, self.out_float)

        #separate 32 and 64
        #remove empty
        temp_hdf5_names = []
        temp_hdf5_gridsizes = []
        if self.train:
            if (not self.is_undc) and self.out_float:
                for name in self.hdf5_names:
                    hdf5_file = h5py.File(self.data_dir+"/"+name+".hdf5", 'r')
                    for grid_size in [32,64]:
                        float_grid = hdf5_file[str(grid_size)+"_float"][:]
                        if np.any(float_grid>=0):
                            temp_hdf5_names.append(name)
                            temp_hdf5_gridsizes.append(grid_size)
            else:
                for name in self.hdf5_names:
                    for grid_size in [32,64]:
                        temp_hdf5_names.append(name)
                        temp_hdf5_gridsizes.append(grid_size)
        else:
            for name in self.hdf5_names:
                temp_hdf5_names.append(name)
                temp_hdf5_gridsizes.append(self.output_grid_size)

        self.hdf5_names = temp_hdf5_names
        self.hdf5_gridsizes = temp_hdf5_gridsizes
        print("Non-trivial Total#", len(self.hdf5_names), self.input_type, self.out_bool, self.out_float)

    def __len__(self):
        return len(self.hdf5_names)

    def __getitem__(self, index):
        hdf5_dir = self.data_dir+"/"+self.hdf5_names[index]+".hdf5"
        grid_size = self.hdf5_gridsizes[index]

        if self.train:
            if self.input_type=="voxel":
                if self.is_undc:
                    gt_output_bool_,gt_output_float_,gt_input_ = read_and_augment_data_undc(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,aug_permutation=True,aug_reversal=True,aug_inversion=False)
                else:
                    gt_output_bool_,gt_output_float_,gt_input_ = read_and_augment_data_ndc(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,aug_permutation=True,aug_reversal=True,aug_inversion=False)
            elif self.input_type=="sdf" or self.input_type=="udf":
                if self.is_undc:
                    gt_output_bool_,gt_output_float_,gt_input_ = read_and_augment_data_undc(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,aug_permutation=True,aug_reversal=True,aug_inversion=True)
                else:
                    gt_output_bool_,gt_output_float_,gt_input_ = read_and_augment_data_ndc(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,aug_permutation=True,aug_reversal=True,aug_inversion=True)
        else:
            if self.input_only:
                gt_output_bool_,gt_output_float_,gt_input_ = read_data_input_only(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,self.is_undc)
            else:
                gt_output_bool_,gt_output_float_,gt_input_ = read_data(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,self.is_undc)

        if self.input_type=="udf":
            gt_input_ = np.abs(gt_input_)


        if self.out_bool:
            gt_output_bool_ = np.transpose(gt_output_bool_, [3,0,1,2]).astype(np.float32)
            gt_output_bool_mask_ = np.zeros(gt_output_bool_.shape, np.float32)

            if self.is_undc:
                if self.input_type=="sdf" or self.input_type=="udf":
                    tmp_mask = ( (gt_input_>-1) & (gt_input_<1) )
                    gt_output_bool_mask_[0,:-1,:,:] = tmp_mask[:-1,:,:] & tmp_mask[1:,:,:]
                    gt_output_bool_mask_[1,:,:-1,:] = tmp_mask[:,:-1,:] & tmp_mask[:,1:,:]
                    gt_output_bool_mask_[2,:,:,:-1] = tmp_mask[:,:,:-1] & tmp_mask[:,:,1:]
                elif self.input_type=="voxel":
                    tmp_mask = (gt_input_!=gt_input_[0,0,0])
                    gt_output_bool_mask_[0,:,1:,1:] = tmp_mask[:,:-1,:-1] & tmp_mask[:,1:,:-1] & tmp_mask[:,:-1,1:] & tmp_mask[:,1:,1:]
                    gt_output_bool_mask_[1,1:,:,1:] = tmp_mask[:-1,:,:-1] & tmp_mask[1:,:,:-1] & tmp_mask[:-1,:,1:] & tmp_mask[1:,:,1:]
                    gt_output_bool_mask_[2,1:,1:,:] = tmp_mask[:-1,:-1,:] & tmp_mask[1:,:-1,:] & tmp_mask[:-1,1:,:] & tmp_mask[1:,1:,:]
            else:
                if self.input_type=="voxel":
                    tmp_mask = np.zeros([grid_size-1,grid_size-1,grid_size-1], np.uint8)
                    gt_input_pos = (gt_input_!=gt_input_[0,0,0])
                    gt_input_neg = (gt_input_==gt_input_[0,0,0])
                    for i in [-1,0,1]:
                        for j in [-1,0,1]:
                            for k in [-1,0,1]:
                                tmp_mask = tmp_mask | gt_input_neg[1+i:grid_size+i,1+j:grid_size+j,1+k:grid_size+k]
                    tmp_mask = tmp_mask & gt_input_pos[1:-1,1:-1,1:-1]
                    for i in [0,1]:
                        for j in [0,1]:
                            for k in [0,1]:
                                gt_output_bool_mask_[0,1+i:grid_size+i,1+j:grid_size+j,1+k:grid_size+k] = np.maximum(gt_output_bool_mask_[0,1+i:grid_size+i,1+j:grid_size+j,1+k:grid_size+k], tmp_mask)

        if self.out_float:
            gt_output_float_ = np.transpose(gt_output_float_, [3,0,1,2])
            gt_output_float_mask_ = (gt_output_float_>=0).astype(np.float32)

        gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)

        #crop to save space & time
        #get bounding box
        if self.train:
            if not self.out_float:
                valid_flag = gt_output_bool_mask_[0]
            elif not self.out_bool:
                valid_flag = np.max(gt_output_float_mask_,axis=0)
            else:
                valid_flag = gt_output_bool_mask_[0] | np.max(gt_output_float_mask_,axis=0)

            #x
            ray = np.max(valid_flag,(1,2))
            xmin = -1
            xmax = -1
            for i in range(grid_size+1):
                if ray[i]>0:
                    if xmin==-1:
                        xmin = i
                    xmax = i
            #y
            ray = np.max(valid_flag,(0,2))
            ymin = -1
            ymax = -1
            for i in range(grid_size+1):
                if ray[i]>0:
                    if ymin==-1:
                        ymin = i
                    ymax = i
            #z
            ray = np.max(valid_flag,(0,1))
            zmin = -1
            zmax = -1
            for i in range(grid_size+1):
                if ray[i]>0:
                    if zmin==-1:
                        zmin = i
                    zmax = i

            xmax += 1
            ymax += 1
            zmax += 1

        else:
            xmin = 0
            xmax = grid_size+1
            ymin = 0
            ymax = grid_size+1
            zmin = 0
            zmax = grid_size+1

        if self.out_bool:
            gt_output_bool = gt_output_bool_[:,xmin:xmax,ymin:ymax,zmin:zmax]
            gt_output_bool_mask = gt_output_bool_mask_[:,xmin:xmax,ymin:ymax,zmin:zmax]
        if self.out_float:
            gt_output_float = gt_output_float_[:,xmin:xmax,ymin:ymax,zmin:zmax]
            gt_output_float_mask = gt_output_float_mask_[:,xmin:xmax,ymin:ymax,zmin:zmax]

        xmin = xmin-self.receptive_padding
        xmax = xmax+self.receptive_padding
        ymin = ymin-self.receptive_padding
        ymax = ymax+self.receptive_padding
        zmin = zmin-self.receptive_padding
        zmax = zmax+self.receptive_padding

        xmin_pad = 0
        xmax_pad = xmax-xmin
        ymin_pad = 0
        ymax_pad = ymax-ymin
        zmin_pad = 0
        zmax_pad = zmax-zmin
        if self.input_type=="sdf" or self.input_type=="udf":
            if gt_input_[0,0,0,0]>0:
                gt_input = np.full([1,xmax_pad,ymax_pad,zmax_pad],10,np.float32)
            else:
                gt_input = np.full([1,xmax_pad,ymax_pad,zmax_pad],-10,np.float32)
        elif self.input_type=="voxel":
            if gt_input_[0,0,0,0]>0:
                gt_input = np.full([1,xmax_pad,ymax_pad,zmax_pad],1,np.float32)
            else:
                gt_input = np.full([1,xmax_pad,ymax_pad,zmax_pad],0,np.float32)

        if xmin<0:
            xmin_pad -= xmin
            xmin = 0
        if xmax>grid_size+1:
            xmax_pad += (grid_size+1-xmax)
            xmax=grid_size+1
        if ymin<0:
            ymin_pad -= ymin
            ymin = 0
        if ymax>grid_size+1:
            ymax_pad += (grid_size+1-ymax)
            ymax=grid_size+1
        if zmin<0:
            zmin_pad -= zmin
            zmin = 0
        if zmax>grid_size+1:
            zmax_pad += (grid_size+1-zmax)
            zmax=grid_size+1

        gt_input[:,xmin_pad:xmax_pad,ymin_pad:ymax_pad,zmin_pad:zmax_pad] = gt_input_[:,xmin:xmax,ymin:ymax,zmin:zmax]


        #the current code assumes that each cell in the input is a unit cube
        #clip to ignore far-away cells
        gt_input = np.clip(gt_input, -2, 2)

        # #if you want to relax the unit-cube assumption, comment out the above clipping code, and uncomment the following code
        # if self.train and self.input_type=="sdf":
        #     if np.random.randint(2)==0:
        #         gt_input = gt_input * (np.random.random()*2+0.001)
        #     else:
        #         gt_input = gt_input * 10**(-np.random.random()*3)


        if self.out_bool and self.out_float:
            return gt_input, gt_output_bool, gt_output_bool_mask, gt_output_float, gt_output_float_mask
        elif self.out_bool:
            return gt_input, gt_output_bool, gt_output_bool_mask
        elif self.out_float:
            return gt_input, gt_output_float, gt_output_float_mask




#only for testing
class single_shape_grid(torch.utils.data.Dataset):
    def __init__(self, data_dir, receptive_padding, input_type, is_undc):
        self.data_dir = data_dir
        self.receptive_padding = receptive_padding
        self.input_type = input_type
        self.is_undc = is_undc

    def __len__(self):
        return 1

    def __getitem__(self, index):

        if self.input_type=="sdf" or self.input_type=="udf":
            if self.data_dir.split(".")[-1]=="sdf":
                LOD_input = read_sdf_file_as_3d_array(self.data_dir)
            elif self.data_dir.split(".")[-1]=="hdf5":
                grid_size = 64
                hdf5_file = h5py.File(self.data_dir, 'r')
                LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
                hdf5_file.close()
            else:
                print("ERROR: invalid input type - only support sdf or hdf5")
                exit(-1)

            if LOD_input.shape[1]!=LOD_input.shape[0] or LOD_input.shape[2]!=LOD_input.shape[0]:
                print("ERROR: the code only supports grids with dimx=dimy=dimz, but the given grid is", LOD_input.shape)
                exit(-1)

            grid_size = LOD_input.shape[0]-1
            gt_input_ = LOD_input*grid_size #denormalize

            if self.input_type=="udf":
                gt_input_ = np.abs(gt_input_)

        elif self.input_type=="voxel":
            if self.data_dir.split(".")[-1]=="binvox":
                LOD_input = read_binvox_file_as_3d_array(self.data_dir)
            elif self.data_dir.split(".")[-1]=="hdf5":
                grid_size = 64
                hdf5_file = h5py.File(self.data_dir, 'r')
                LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
                hdf5_file.close()
            else:
                print("ERROR: invalid input type - only support binvox or hdf5")
                exit(-1)

            if LOD_input.shape[1]!=LOD_input.shape[0] or LOD_input.shape[2]!=LOD_input.shape[0]:
                print("ERROR: the code only supports grids with dimx=dimy=dimz, but the given grid is", LOD_input.shape)
                exit(-1)

            grid_size = LOD_input.shape[0]+2
            #add padding
            gt_input_ = np.zeros([grid_size,grid_size,grid_size], np.uint8)
            gt_input_[1:-1,1:-1,1:-1] = LOD_input
            grid_size = grid_size-1


        #prepare mask
        if self.is_undc:
            gt_output_bool_mask_ = np.zeros([3,grid_size+1,grid_size+1,grid_size+1], np.float32)
            if self.input_type=="sdf" or self.input_type=="udf":
                tmp_mask = ( (gt_input_>-1) & (gt_input_<1) )
                gt_output_bool_mask_[0,:-1,:,:] = tmp_mask[:-1,:,:] & tmp_mask[1:,:,:]
                gt_output_bool_mask_[1,:,:-1,:] = tmp_mask[:,:-1,:] & tmp_mask[:,1:,:]
                gt_output_bool_mask_[2,:,:,:-1] = tmp_mask[:,:,:-1] & tmp_mask[:,:,1:]
            elif self.input_type=="voxel":
                tmp_mask = (gt_input_!=gt_input_[0,0,0])
                gt_output_bool_mask_[0,:,1:,1:] = tmp_mask[:,:-1,:-1] & tmp_mask[:,1:,:-1] & tmp_mask[:,:-1,1:] & tmp_mask[:,1:,1:]
                gt_output_bool_mask_[1,1:,:,1:] = tmp_mask[:-1,:,:-1] & tmp_mask[1:,:,:-1] & tmp_mask[:-1,:,1:] & tmp_mask[1:,:,1:]
                gt_output_bool_mask_[2,1:,1:,:] = tmp_mask[:-1,:-1,:] & tmp_mask[1:,:-1,:] & tmp_mask[:-1,1:,:] & tmp_mask[1:,1:,:]
        else:
            gt_output_bool_mask_ = np.zeros([1,grid_size+1,grid_size+1,grid_size+1], np.float32)
            if self.input_type=="voxel":
                tmp_mask = np.zeros([grid_size-1,grid_size-1,grid_size-1], np.uint8)
                gt_input_pos = (gt_input_!=gt_input_[0,0,0])
                gt_input_neg = (gt_input_==gt_input_[0,0,0])
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        for k in [-1,0,1]:
                            tmp_mask = tmp_mask | gt_input_neg[1+i:grid_size+i,1+j:grid_size+j,1+k:grid_size+k]
                tmp_mask = tmp_mask & gt_input_pos[1:-1,1:-1,1:-1]
                for i in [0,1]:
                    for j in [0,1]:
                        for k in [0,1]:
                            gt_output_bool_mask_[0,1+i:grid_size+i,1+j:grid_size+j,1+k:grid_size+k] = np.maximum(gt_output_bool_mask_[0,1+i:grid_size+i,1+j:grid_size+j,1+k:grid_size+k], tmp_mask)

        gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)

        gt_output_bool_mask = gt_output_bool_mask_

        #receptive field padding
        padded = grid_size+1+self.receptive_padding*2
        
        if self.input_type=="sdf" or self.input_type=="udf":
            if gt_input_[0,0,0,0]>0:
                gt_input = np.full([1,padded,padded,padded],10,np.float32)
            else:
                gt_input = np.full([1,padded,padded,padded],-10,np.float32)
        elif self.input_type=="voxel":
            if gt_input_[0,0,0,0]>0:
                gt_input = np.full([1,padded,padded,padded],1,np.float32)
            else:
                gt_input = np.full([1,padded,padded,padded],0,np.float32)

        gt_input[:,self.receptive_padding:-self.receptive_padding,self.receptive_padding:-self.receptive_padding,self.receptive_padding:-self.receptive_padding] = gt_input_


        #the current code assumes that each cell in the input is a unit cube
        #clip to ignore far-away cells
        gt_input = np.clip(gt_input, -2, 2)

        return gt_input, gt_output_bool_mask
