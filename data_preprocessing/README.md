# data_preprocessing
Tools for pre-processing raw mesh data and converting them to the representation used for training.

## Requirements
- Ubuntu
- Python 3 with numpy, h5py and Cython

Build Cython module in each subfolder:
```
python setup.py build_ext --inplace
```


## Usage

You can download [ABC](https://deep-geometry.github.io/abc-dataset/) and [Thingi10K](https://ten-thousand-models.appspot.com/) from their websites. Or you could prepare your own dataset. All the shapes need to be closed triangle meshes and in obj format (so ShapeNet is not an option...). In the following, I would assume you are using [ABC](https://deep-geometry.github.io/abc-dataset/) dataset.

Run *simplify_obj.py* to normalize the meshes and remove empty folders.

Go to folder *get_groundtruth_NDC*. Then run *get_gt_LOD.py* to get training data.


## Removing invalid shapes

We use the same training shapes as those in [NMC](https://github.com/czq142857/NMC).
The shapes used in our experiments are recorded in *abc_obj_list.txt* and *thingi10k_obj_list.txt*.


## Executable files IntersectionXYZpn

See [data_utils](https://github.com/czq142857/NMC/tree/master/data_utils) from [NMC](https://github.com/czq142857/NMC).


