# NDC
PyTorch implementation of Neural Dual Contouring.

<img src='teaser.png' />


## Citation
We are still writing the paper while adding more improvements and applications.
If you find our work useful in your research, please consider citing our prior work [Neural Marching Cubes (NMC)](https://github.com/czq142857/NMC).

	@article{chen2021nmc,
	  title={Neural Marching Cubes},
	  author={Zhiqin Chen and Hao Zhang},
	  journal={arXiv preprint arXiv:2106.11272},
	  year={2021}
	}




## Requirements
- Python 3 with numpy, h5py, scipy, sklearn and Cython
- [PyTorch 1.8](https://pytorch.org/get-started/locally/) (other versions may also work)

Build Cython module:
```
python setup.py build_ext --inplace
```


## Datasets and pre-trained weights
For data preparation, please see [data_preprocessing](https://github.com/czq142857/NDC/tree/master/data_preprocessing).

We provide the ready-to-use datasets here.

- [groundtruth_NDC.7z](https://drive.google.com/file/d/1vBisjHln8NUtbjHjcF-tcYimRDJNZ8Xo/view?usp=sharing)

Backup links:

- [groundtruth_NDC.7z](https://pan.baidu.com/s/13ICHqjYc3FOZvzF56dycJw) (pwd: 1234)

We also provide the pre-trained network weights in this repo.
- *network_float.pth* in the main directory is for SDF inputs.
- weights in folder *weights_for_voxel_input* are for voxel inputs.


## Training and Testing

To train/test NDC with SDF input:
```
python main.py --train_float --epoch 400 --data_dir groundtruth/gt_NDC --input_type sdf
python main.py --test_bool_float --data_dir groundtruth/gt_NDC --input_type sdf
```

To train/test NDC with SDF input:
```
python main.py --train_bool --epoch 400 --data_dir groundtruth/gt_NDC --input_type voxel
python main.py --train_float --epoch 400 --data_dir groundtruth/gt_NDC --input_type voxel
python main.py --test_bool_float --data_dir groundtruth/gt_NDC --input_type voxel
```

To evaluate Chamfer Distance, Normal Consistency, F-score, Edge Chamfer Distance, Edge F-score, you need to have the ground truth normalized obj files ready in a folder *objs*. See [data_preprocessing](https://github.com/czq142857/NDC/tree/master/data_preprocessing) for how to prepare the obj files. Then you can run:
```
python eval_cd_nc_f1_ecd_ef1.py
```

To count the number of triangles and vertices, run:
```
python eval_v_t_count.py
```

If you want to test on your own dataset, please refer to [data_preprocessing](https://github.com/czq142857/NDC/tree/master/data_preprocessing) for how to convert obj files into SDF grids and voxel grids. If your data are not meshes (say your data are already voxel grids), you can modify the code in *utils.py* to read your own data format. Check function *read_data_input_only* in *utils.py* for an example.

