# NDC
PyTorch implementation for paper [Neural Dual Contouring](https://arxiv.org/abs/2202.01999), [Zhiqin Chen](https://czq142857.github.io/), [Andrea Tagliasacchi](https://taiya.github.io/), [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/), [Hao Zhang](http://www.cs.sfu.ca/~haoz/). Note: this is a partial implementation. The full implementation will be released here after the paper is published.

### [Paper](https://arxiv.org/abs/2202.01999)

<img src='teaser.png' />

## Citation
If you find our work useful in your research, please consider citing:

	@misc{chen2022ndc,
	  title={Neural Dual Contouring}, 
	  author={Zhiqin Chen and Andrea Tagliasacchi and Thomas Funkhouser and Hao Zhang},
	  year={2022},
	  eprint={2202.01999},
	  archivePrefix={arXiv},
	  primaryClass={cs.CV}
	}



## Requirements
- Python 3 with numpy, h5py, scipy and Cython
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

To train/test NDC with voxel input:
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

