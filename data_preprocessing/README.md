# data_preprocessing
Tools for pre-processing raw mesh data and converting them to the representation used for training.

## Requirements
- Ubuntu
- Python 3 with numpy, h5py, scipy, trimesh, and Cython

Build Cython module in each subfolder:
```
python setup.py build_ext --inplace
```


## Usage

You can download [ABC](https://deep-geometry.github.io/abc-dataset/), [Thingi10K](https://ten-thousand-models.appspot.com/), [FAUST](http://faust.is.tue.mpg.de/), and [MGN](http://virtualhumans.mpi-inf.mpg.de/mgn/) from their websites. Or you could prepare your own dataset.

For [Matterport3D](https://niessner.github.io/Matterport/), I do not have the code for converting the raw sensor data into point clouds, and I am not allowed to distribute the already processed data. So if you have questions regarding [Matterport3D](https://niessner.github.io/Matterport/), please contact the [Matterport3D](https://niessner.github.io/Matterport/) team.

All the shapes used for training NDC need to be closed triangle meshes and in obj format, so [ABC](https://deep-geometry.github.io/abc-dataset/) is an ideal dataset. Training UNDC does not require closed shapes, but we never tried training on other datasets. In the following, I would assume you are using [ABC](https://deep-geometry.github.io/abc-dataset/) dataset.

We use the same training shapes as those in [NMC](https://github.com/czq142857/NMC).
The shapes used in our experiments are recorded in *abc_obj_list.txt*.

Run *simplify_obj.py* to normalize the meshes and remove empty folders.

Go to folder *get_groundtruth_NDC*. Then run *get_gt_LOD.py* to get training data for NDC.

Go to folder *get_groundtruth_UNDC*. Then run *get_gt_LOD.py* to get training data for UNDC.

When training on noisy point cloud inputs, it is better to perform data augmentation, as follows.

Run *augment_obj.py* to write augmented shapes into the data folder. It only performs scaling and translation augmentation.

Go to folder *get_groundtruth_UNDC_augmented*. Then run *get_gt_LOD.py* to get training data for UNDC.


## Executable files IntersectionXYZpn, SDFGen, and VOXGen

See [data_utils](https://github.com/czq142857/NMC/tree/master/data_utils) from [NMC](https://github.com/czq142857/NMC).


