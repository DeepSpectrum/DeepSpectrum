**DeepSpectrum** is a Python toolkit for feature extraction from audio data with pre-trained Image Convolutional Neural Networks (CNNs). It features an extraction pipeline which first creates visual representations for audio data - plots of spectrograms or chromagrams - and then feeds them to a pre-trained Image CNN. Activations of a specific layer then form the final feature vectors.

**(c) 2017-2018 Shahin Amiriparian, Maurice Gercuk, Sandra Ottl, Björn Schuller: Universität Augsburg**
Published under GPLv3, see the LICENSE.md file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at tum.de) or Maurice Gercuk (gerczuk at fim.uni-passau.de).

# Citing
If you use DeepSpectrum or any code from DeepSpectrum in your research work, you are kindly asked to acknowledge the use of DeepSpectrum in your publications.
> S. Amiriparian, M. Gerczuk, S. Ottl, N. Cummins, M. Freitag, S. Pugachevskiy, A. Baird and B. Schuller. Snore Sound Classification using Image-Based Deep Spectrum Features. In Proceedings of INTERSPEECH (Vol. 17, pp. 2017-434)


# Installation
This program supports pipenv for dependency resolution and installation and we highly recommend you to use it. In addition to the actual tool in `deep-spectrum` there is also another tool which helps with aquiring the pre-trained AlexNet model and converting it to a tensorflow compatible format. This relies on the `caffe-tensorflow` conversion tool found at https://github.com/ethereon/caffe-tensorflow 

## Dependencies
* Python 3.6 with pipenv for the Deep Spectrum tool (`pip install pipenv`)
* Python 2.7 to download and convert the AlexNet model

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

## Configuration
If you used the included script to download the AlexNet model, the tool is already configured correctly for usage. Otherwise, you have to adjust your configuration file. The default file can be found in `deep-spectrum/deep_spectrum_extractor/cli/deep.conf`:
```
[main]
size = 227
gpu = 1
backend = tensorflow

[tensorflow-nets]
alexnet = AlexNet.pb
```
Under `tensorflow-nets` you can define network names and their corresponding model files (in .pb format). Currently, only the converted AlexNet model is officially supported. You can try it with different models but might have to adjust code in `deep-spectrum/deep_spectrum_extractor/backend/extractor.py` in order to correctly identify your model's layer outputs in the graph's tensors. We plan on including a extraction backend for the new TensorFlow Hub (https://www.tensorflow.org/hub/) in the near future to make using different models easier.



