#!/usr/bin/env python
import re
import sys
from setuptools import setup, find_packages
from subprocess import CalledProcessError, check_output

PROJECT = "DeepSpectrum"
VERSION = "0.6.6"
LICENSE = "GPLv3+"
AUTHOR = "Maurice Gerczuk"
AUTHOR_EMAIL = "gerczuk@fim.uni-passau.de"
URL = 'https://github.com/DeepSpectrum/DeepSpectrum'

install_requires = [
    "audeep>=0.9.4",
    "imread>=0.7.0",
    "tqdm>=4.30.0",
    "matplotlib>=3.3",
    "numba==0.48.0",
    "librosa>=0.7.0, <0.8.0",
    "click>=7.0",
    "Pillow >=6.0.0",
    "tensorflow-gpu>=1.15.2, <2",
    "opencv-python>=4.0.0.21",
    "torch>=1.2.0",
    "torchvision>=0.5.0"
]


tests_require = ['pytest>=4.4.1', 'pytest-cov>=2.7.1']
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []
packages = find_packages('src')

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=["Any"],
    scripts=[],
    provides=[],
    python_requires=">=3.6, <3.8",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    namespace_packages=[],
    packages=packages,
    package_dir={'': 'src',
                 'audeep': 'auDeep/audeep'},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deepspectrum = deepspectrum.__main__:cli",
        ]
    },
    url=URL,
    zip_safe=False,
)
