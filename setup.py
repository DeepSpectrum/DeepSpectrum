#!/usr/bin/env python
import re
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from setuptools import setup, find_packages
from subprocess import CalledProcessError, check_output

PROJECT = "DeepSpectrum"
VERSION = "0.6.8"
LICENSE = "GPLv3+"
AUTHOR = "Maurice Gerczuk"
AUTHOR_EMAIL = "gerczuk@fim.uni-passau.de"
URL = 'https://github.com/DeepSpectrum/DeepSpectrum'

with open("DESCRIPTION.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

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
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    descrption="DeepSpectrum is a Python toolkit for feature extraction from audio data with pre-trained Image Convolutional Neural Networks (CNNs).",
    platforms=["Any"],
    scripts=[],
    provides=[],
    python_requires="~=3.7.0",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    namespace_packages=[],
    packages=packages,
    package_dir={'': 'src'},
                 #'audeep': 'auDeep/audeep'},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deepspectrum = deepspectrum.__main__:cli",
        ]
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
            
        'Environment :: GPU :: NVIDIA CUDA :: 10.0',
        # Indicate who your project is intended for
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.7',
    ],
    keywords='machine-learning audio-analysis science research',
    project_urls={
        'Source': 'https://github.com/DeepSpectrum/DeepSpectrum/',
        'Tracker': 'https://github.com/DeepSpectrum/DeepSpectrum/issues',
    },
    url=URL,
    zip_safe=False,
)
