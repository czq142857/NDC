from distutils.core import setup
from Cython.Build import cythonize
import numpy
#python setup.py build_ext --inplace
setup(name='cutils', include_dirs = [numpy.get_include()],  ext_modules=cythonize("cutils.pyx"))