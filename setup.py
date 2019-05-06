#!/usr/bin/env python
import re
import sys
from setuptools import setup, find_packages
from subprocess import CalledProcessError, check_output

PROJECT = "DeepSpectrum"
VERSION = "0.3.2"
LICENSE = "GPLv3+"
AUTHOR = "Maurice Gerczuk"
AUTHOR_EMAIL = "gerczuk@fim.uni-passau.de"

dependencies = [
    'numpy>=1.16',
    'scipy>=1.2.0',
    'pandas>=0.24.0',
    'imread>=0.7.0',
    'pysoundfile',
    'tqdm>=4.30.0',
    'matplotlib>=3.0.2',
    'opencv-python>=4.0.0.21',
    'librosa',
    'scikit-learn>=0.20.2',
    'liac-arff>=2.3.1',
    'statsmodels>=0.9',
    'dataclasses>=0.6',
    'click',
    'Pillow'
]


if sys.version_info < (3,7):
    if sys.version_info >= (3,6):
        dependencies.append('dataclasses>=0.6')
    else:
        sys.exit('Python < 3.6 is not supported')

try:
    import tensorflow

    tensorflow_found = True
except ImportError:
    tensorflow_found = False

if not tensorflow_found:
    # inspired by cmake's FindCUDA
    nvcc_version_regex = re.compile("release (?P<major>[0-9]+)\\.(?P<minor>[0-9]+)")
    use_gpu = False

    try:
        output = str(check_output(["nvcc", "--version"]))
        version_string = nvcc_version_regex.search(output)

        if version_string:
            major = int(version_string.group("major"))
            minor = int(version_string.group("minor"))
            if major == 10 and minor == 0:
                print("detected compatible CUDA version %d.%d" % (major, minor))
                dependencies.append("tensorflow-gpu>=1.13.0")
                use_gpu=True

            if major == 9:
                print("detected compatible CUDA version %d.%d" % (major, minor))
                dependencies.append("tensorflow-gpu==1.12.0")
                use_gpu=True

            else:
                print("detected incompatible CUDA version %d.%d" % (major, minor))

        else:
            print("CUDA detected, but unable to parse version")
    except CalledProcessError:
        print("no CUDA detected")
    except Exception as e:
        print("error during CUDA detection: %s", e)

    if not use_gpu:
        dependencies.append("tensorflow>=1.13.0")
else:
    print("tensorflow already installed, skipping CUDA detection")

setup(
    name=PROJECT,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=["Any"],
    scripts=[],
    provides=[],
    python_requires='>=3.6',
    install_requires=dependencies,
    namespace_packages=[],
    packages=find_packages(),
    include_package_data=True,
    entry_points = {
                   'console_scripts': [
                       'ds-features = deepspectrum.cli.ds_features:main',
                       'ds-reduce = deepspectrum.tools.feature_reduction:main',
                       'ds-scikit = deepspectrum.cli.ds_scikit:main',
                       'ds-dnn = deepspectrum.learn.tf.dnn.__main__:main',
                       'ds-rnn = deepspectrum.learn.tf.rnn.__main__:main',
                       'ds-cm = deepspectrum.tools.performance_stats:main',
                       'ds-image-features = deepspectrum.cli.image_features:main',
                       'ds-plot = deepspectrum.cli.ds_plot:main',
                       'ds-help = deepspectrum.cli.ds_help:main',
                       'ds-results = deepspectrum.cli.ds_results:main',
                       'deepspectrum = deepspectrum.__main__:cli'
                   ]
               },
    zip_safe = False
)
