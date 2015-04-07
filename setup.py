from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'onlinehdp',
  ext_modules = cythonize("onlinehdp.pyx"),
  include_dirs = [numpy.get_include()]
)
