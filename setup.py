#!/usr/bin/env python
import re
import sys
from setuptools import setup, find_packages
from subprocess import CalledProcessError, check_output

PROJECT = "DeepSpectrum"
VERSION = "0.6.4"
LICENSE = "GPLv3+"
AUTHOR = "Maurice Gerczuk"
AUTHOR_EMAIL = "gerczuk@fim.uni-passau.de"
URL = 'https://github.com/DeepSpectrum/DeepSpectrum'

install_requires = [
    "numpy>=1.16",
    "scipy>=1.2.0",
    "imread>=0.7.0",
    "tqdm>=4.30.0",
    "matplotlib>=3.0.2",
    "librosa>=0.6.4",
    "click>=7.0",
    "Pillow >=6.0.0, <7",
    "xarray"
]

try:
    import cv2
    cv2_found = True
except ImportError:
    cv2_found = False

if not cv2_found:
    install_requires.append("opencv-python>=4.0.0.21")

try:
    import torch
    torch_found = True
except ImportError:
    torch_found = False

if not torch_found:
    install_requires.append("torch>=1.1.0")
    install_requires.append("torchvision>=0.3.0")

try:
    import tensorflow

    tensorflow_found = True
except ImportError:
    tensorflow_found = False

if not tensorflow_found:
    # inspired by cmake's FindCUDA
    nvcc_version_regex = re.compile(
        "release (?P<major>[0-9]+)\\.(?P<minor>[0-9]+)")
    use_gpu = False

    try:
        output = str(check_output(["nvcc", "--version"]))
        version_string = nvcc_version_regex.search(output)

        if version_string:
            major = int(version_string.group("major"))
            minor = int(version_string.group("minor"))
            if major == 10 and minor == 0:
                print("detected compatible CUDA version %d.%d" %
                      (major, minor))
                install_requires.append("tensorflow-gpu >=1.13.0, <2")
                use_gpu = True

            if major == 9:
                print("detected compatible CUDA version %d.%d" %
                      (major, minor))
                install_requires.append("tensorflow-gpu==1.12.0")
                use_gpu = True

            else:
                print("detected incompatible CUDA version %d.%d" %
                      (major, minor))

        else:
            print("CUDA detected, but unable to parse version")
    except CalledProcessError:
        print("no CUDA detected")
    except Exception as e:
        print("error during CUDA detection: %s", e)

    if not use_gpu:
        install_requires.append("tensorflow >=1.13.0, <2")
else:
    pass

tests_require = ['pytest>=4.4.1', 'pytest-cov>=2.7.1']
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=["Any"],
    scripts=[],
    provides=[],
    python_requires=">=3.6",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    namespace_packages=[],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deepspectrum = deepspectrum.__main__:cli",
        ]
    },
    url=URL,
    zip_safe=False,
)
