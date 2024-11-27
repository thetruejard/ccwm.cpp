# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.

ext_modules = [
    Pybind11Extension(
        "ccwm_cpp",
        sorted(glob('src/*.cpp', recursive=True)),
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=17,
    ),
]

setup(
    name="ccwm_cpp",
    version=__version__,
    author="Jared Watrous",
    author_email="jwatrous2002@gmail.com",
    url="https://github.com/thetruejard/ccwm.cpp",
    description="A C++ implementation and Python interface for CCWM",
    long_description="",
    ext_modules=ext_modules,
    #extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)