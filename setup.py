from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

sourcefiles = ['deriv_utils.pyx', 'glm.pyx', 'c_utils.c']

extensions = [Extension("glm", sourcefiles)]

setup(
  name = 'onlinehdp',
  ext_modules = cythonize(extensions),
  include_dirs = [numpy.get_include()]
)
