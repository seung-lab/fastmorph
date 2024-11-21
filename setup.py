import os
import setuptools
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++17', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++17', '-O3'
  ]

setuptools.setup(
  name="fastmorph",
  version="1.2.1",
  setup_requires=["numpy","pybind11"],
  install_requires=['numpy', 'edt', 'fill-voids', 'connected-components-3d', 'fastremap'],
  python_requires=">=3.8.0", # >= 3.8 < 4.0
  author="William Silversmith",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'fastmorph': [
      'LICENSE',
    ],
  },
  ext_modules=[
    Pybind11Extension(
        "fastmorphops",
        ["fastmorph/fastmorphops.cpp"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
  ],
  description="Morphological image processing for 3D multi-label images.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  keywords = "morphological dialate erode dilation erosion close open fill image processing",
  url = "https://github.com/seung-lab/fastmorph/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ],  
)


