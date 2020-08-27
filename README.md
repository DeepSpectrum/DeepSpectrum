![Codecov](https://img.shields.io/codecov/c/github/deepspectrum/deepspectrum?style=flat)
![CI status](https://github.com/deepspectrum/deepspectrum/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/DeepSpectrum.svg)](https://badge.fury.io/py/DeepSpectrum)
![PyPI - License](https://img.shields.io/pypi/l/DeepSpectrum)

**DeepSpectrum** is a Python toolkit for feature extraction from audio data with pre-trained Image Convolutional Neural Networks (CNNs). It features an extraction pipeline which first creates visual representations for audio data - plots of spectrograms or chromagrams - and then feeds them to a pre-trained Image CNN. Activations of a specific layer then form the final feature vectors.

**(c) 2017-2020 Shahin Amiriparian, Maurice Gerczuk, Sandra Ottl, Björn Schuller: Universität Augsburg**
Published under GPLv3, see the LICENSE.md file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at tum.de) or Maurice Gercuk (maurice.gerczuk at informatik.uni-augsburg.de).

# Citing
If you use DeepSpectrum or any code from DeepSpectrum in your research work, you are kindly asked to acknowledge the use of DeepSpectrum in your publications.
> S. Amiriparian, M. Gerczuk, S. Ottl, N. Cummins, M. Freitag, S. Pugachevskiy, A. Baird and B. Schuller. Snore Sound Classification using Image-Based Deep Spectrum Features. In Proceedings of INTERSPEECH (Vol. 17, pp. 2017-434)


# Installation
The easiest way to install DeepSpectrum is through the official pypi package which is built for every release tag on the master branch. For installing different branches or a more manual approach, you can also use the setup.py script with [pip](#installation-through-pip) (only for Linux) and also an environment.yml for installing through [conda](#conda-installation) (recommended on Windows and OSX).



## Dependencies (only for installation with pip)
* Python 3.7
* ffmpeg

## Installing the python package
We recommend that you first setup and activate a virtual python 3.7 environment. Then you can install the toolkit via pip:
```bash
pip install deepspectrum
```
Installation is now complete - you can skip to [configuration](#configuration) or [usage](#using-the-tool).


## Manual Conda installation
You can use the included environment.yml file to create a new virtual python environment with DeepSpectrum by running:
```bash
conda env create -f environment.yml
```
Then activate the environmnet with:
```bash
conda activate DeepSpectrum
```

Installation is now completed - you can skip to [configuration](#configuration) or [usage](#using-the-tool).


## Installation through pip (for Linux)
We recommend that you install the DeepSpectrum tool into a virtual environment. To do so first create a new virtualenvironment:
```bash
virtualenv -p python3 ds_virtualenv
```

This creates a minimal python installation in the folder "ds_virtualenv". You can choose a different name instead of "ds_virtualenv" if you like, but the guide assumes this name.
You can then activate the virtualenv (Linux):
```bash
source ds_virtualenv/bin/activate
```

Once the virtualenv is activated, the tool can be installed from the source directory (containing setup.py) with this command:
```bash
pip install .
```

Installation is now completed - you can skip to [configuration](#configuration) or [usage](#using-the-tool).

## GPU support
DeepSpectrum uses Tensorflow 1.15.2. GPU support should be automatically available, as long as you have CUDA version 10.0. If you cannot install cuda 10.0 globally, you can use Anaconda to install it in a virtual environment along DeepSpectrum.


## Configuration
If you just want to start working with ImageNet pretrained keras-application models, skip to [usage](#using-the-tool). Otherwise, you can adjust your configuration file to use other weights for the supported models. The default file can be found in `deep-spectrum/src/cli/deep.conf`:
```
[main]
size = 227
backend = keras

[keras-nets]
vgg16 = imagenet
vgg19 = imagenet
resnet50 = imagenet
inception_resnet_v2 = imagenet
xception = imagenet
densenet121 = imagenet
densenet169 = imagenet
densenet201 = imagenet
mobilenet= imagenet
mobilenet_v2 = imagenet
nasnet_large = imagenet
nasnet_mobile = imagenet

[pytorch-nets]
alexnet=
squeezenet=
googlenet=

```
Under `keras-nets` you can define network weights for the supported models. Setting the weights for a model to `imagenet` is the default and uses ImageNet pretrained models from `keras-aplications`. Three additional networks are also supported through pytorch: `alexnet`, `squeezenet` and `googlenet`. For these, no definition of the used weights is needed (or possible, for the time being). The downloaded `keras-nets` will be stored in `$HOME/.keras`.

# Using the tool
You can access the scripts provided by the tool from the virtualenvironment by calling `deepspectrum`. The feature extraction component is provided by the subcommand `features`.

## Features for AVEC2018 CES
The command below extracts features from overlapping 1 second windows spaced with a hop size of 0.1 seconds (`-t 1 0.1`) of the the file `Train_DE_01.wav`. It plots mel spectrograms (`-m mel`) and feeds them to a pre-trained VGG16 model (`-en vgg16`). The activations on the fc2 layer (`-fl fc2`) are finally written to `Train_DE_01.arff` as feature vectors in arff format. `-nl` suppresses writing any labels to the output file. The first argument after `deepspectrum features` must be the path to the audiofile(s).
```bash
deepspectrum features Train_DE_01.wav -t 1 0.1 -nl -en vgg16 -fl fc2 -m mel -o Train_DE_01.arff
```

## Commandline Options
All options can also be displayed using `deepspectrum features --help`.
### Required options
| Option   | Description | Default |
|----------|-------------|---------|
| -o, --output | The location of the output feature file. Supported output formats are: Comma separated value files and arff files. If the specified output file's extension is *.arff*, arff is chosen as format, otherwise the output will be in comma separated value format. | None |


### Extracting features from audio chunks
| Option   | Description | Default |
|----------|-------------|---------|
| -t, --window-size-and-hop | Define window and hopsize for feature extraction. E.g `-t 1 0.5` extracts features from 1 second chunks every 0.5 seconds. | Extract from the whole audio file. |
| -s, --start | Set a start time (in seconds) from which features should be extracted from the audio files. | 0 |
| -e, --end | Set an end time until which features should be extracted from the audio files. | None |

### Setting parameters for the audio plots
| Option   | Description | Default |
|----------|-------------|---------|
| -m, --mode | Type of plot to use in the system (Choose from: 'spectrogram', 'mel', 'chroma'). | spectrogram |
| -fs, --frequency-scale | Scale for the y-axis of the plots used by the system (Choose from: 'linear', 'log' and 'mel'). This is ignored if mode=chroma or mode=mel. (default: linear)
| -fql, --frequency-limit | Specify a limit for the y-axis in the spectrogram plot in frequency. | None |
| -d, --delta | If specified, derivatives of the given order of the selected features are displayed in the plots used by the system. | None |
| -nm, --number-of-melbands | Number of melbands used for computing the melspectrogram. Only takes effect with mode=mel. | 128 |
| -nfft | The length of the FFT window used for creating the spectrograms in number of samples. Consider choosing smaller values when extracting from small segments. | The next power of two from 0.025 x sampling_rate_of_wav |
| -cm, --colour-map | Choose a matplotlib colourmap for creating the spectrogram plots. | viridis |

### Parameters for the feature extractor CNN
| Option   | Description | Default |
|----------|-------------|---------|
| -en, --extraction-network | Choose the net for feature extraction as specified in the config file | alexnet |
| -fl, --feature-layer | Name of the layer from which features should be extracted. | fc2 |

### Defining label information
You can use csv files for label information or explicitly set a fixed label for all input files. If you use csv files, numerical features are supported (e.g. for regression). If you do neither of those, each file is assigned the name of its parent directory as label. This can be useful if your folder structure already represents the class labels, e.g.
```
data                          Base Directory of your data
  ├─── class0                 Directory containing members of 'class0'
  |    └─── instance0.wav     Directory containing members of 'class1'
  ├─── class1                      
  |    └─── instance4.wav     
  |    └─── ...
  └─── class2.py              Directory containing members of 'class2'
       └─── instance20.wav  
```

| Option   | Description | Default |
|----------|-------------|---------|
| -lf, --label-file | Specify a comma separated values file containing labels for each *.wav* file. It has to include a header and the first column must specify the name of the audio file (with extension!) | None |
| -tc, --time-continuous | Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column. | False |
| -el, --explicit-label | Specify a single label that will be used for every input file explicitly. | None |
| -nts, --no-timestamps | Remove timestamps from the output. | Write timestamps in feature file. |
| -nl, --no-labels | Remove labels from the output. | Write labels in feature file. |

### Additional output 
| Option   | Description | Default |
|----------|-------------|---------|
| -so, --spectrogram-out | Specify a folder to save the plots used during extraction as .pngs | None |
| -wo, --wav-out | Convenience function to write the chunks of audio data used in the extraction to the specified folder. | None |

### Configuration and Help
| Option   | Description | Default |
|----------|-------------|---------|
| -np, --number-of-processes | Specify the number of processes used for the extraction. Defaults to the number of available CPU cores | None |
| -c, --config | The path to the configuration file used by the program can be given here. If the file does not exist yet, it is created and filled with standard settings. | deep.conf |
| --help | Show help. | None |


### Extracting CNN-Descriptors from images

The tool also provides a commandline utility for extracting CNN descriptors from image data. It can be accessed through `deepspectrum image-features` with a reduced set of options. As with `deepspectrum features`, the first argument should be a folder containing the input image files (.png or .jpg). The available options are: `-o`, `-c`, `-np`, `-en`, `-fl`, `-bs`, `-lf`, `-el`,`-nl` and `--help`. These function the same as described above for `deepspectrum features`.
