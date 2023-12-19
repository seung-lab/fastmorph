#!/bin/zsh

source ~/.zprofile

function compile_wheel {
	workon fm$1
	pip install oldest-supported-numpy pybind11 setuptools wheel
	pip install build
	python -m build
}

compile_wheel 38
compile_wheel 39
compile_wheel 310
compile_wheel 311
compile_wheel 312
