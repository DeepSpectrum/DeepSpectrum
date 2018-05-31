**DeepSpectrum** is a Python toolkit for feature extraction from audio data with pre-trained Image Convolutional Neural Networks (CNNs). It features an extraction pipeline which first creates visual representations for audio data - plots of spectrograms or chromagrams - and then feeds them to a pre-trained Image CNN. Activations of a specific layer then form the final feature vectors.

**(c) 2017-2018 Shahin Amiriparian, Maurice Gercuk, Sandra Ottl, Björn Schuller: Universität Augsburg**
Published under GPLv3, see the LICENSE.md file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at tum.de) or Maurice Gercuk (gerczuk at fim.uni-passau.de).

# Citing
If you use DeepSpectrum or any code from DeepSpectrum in your research work, you are kindly asked to acknowledge the use of DeepSpectrum in your publications.
> S. Amiriparian, M. Gerczuk, S. Ottl, N. Cummins, M. Freitag, S. Pugachevskiy, A. Baird and B. Schuller. Snore Sound Classification using Image-Based Deep Spectrum Features. In Proceedings of INTERSPEECH (Vol. 17, pp. 2017-434)

# Dependencies
* Python 3.6 with pipenv for the Deep Spectrum tool (`pip install pipenv`)
* Python 2.7 to download and convert the AlexNet model

# Installation

## Download and convert AlexNet model
The Deep Spectrum tool uses the ImageNet pretrained AlexNet model to extract features. To download and convert it to a tensorflow compatible format, a script `download_alexnet.sh` is included. The script performs these general steps:
1. Create a python2 virtual environment with tensorflow in `convert-models/`
2. Clone the caffe-tensorflow repository (https://github.com/ethereon/caffe-tensorflow) that is later used to convert the model
3. Fix incompatibilities of the repository with new tensorflow versions
4. Download the model files from https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
5. Run the conversion script `convert-models/caffe-tensorflow/convert.py` to convert the weights to .npy format
6. Load the model into a tensorflow graph and save it to `.pb` format (`convert_to_pb.py`)
7. Move the file to `deep-spectrum/AlexNet.pb`

## Deep Spectrum tool
Install the Deep Spectrum tool from the `deep-spectrum/` directory with pipenv (which also handles the creation of a virtualenv for you):
```bash
cd deep-spectrum
pipenv --site-packages install
```
If you already have installed a recent version (> 1.5) of tensorflow on your system, the tool is now installed. Otherwise, install tensorflow (version 1.8.0 is tested): 
```bash
pipenv install tensorflow==1.8.0
```
Or the CUDA enabled version:
```bash
pipenv install tensorflow-gpu==1.8.0
```
