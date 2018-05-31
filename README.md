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
If you already have installed a recent version (> 1.5) of tensorflow on your system, continue with the [configuration](##configuration). Otherwise, install tensorflow (version 1.8.0 is tested): 
```bash
pipenv install tensorflow==1.8.0
```
Or the CUDA enabled version:
```bash
pipenv install tensorflow-gpu==1.8.0
```

## Configuration
If you used the included script to download the AlexNet model, the tool is already configured correctly for [usage](#using-the-tool). Otherwise, you have to adjust your configuration file. The default file can be found in `deep-spectrum/deep_spectrum_extractor/cli/deep.conf`:
```
[main]
size = 227
gpu = 1
backend = tensorflow

[tensorflow-nets]
alexnet = AlexNet.pb
```
Under `tensorflow-nets` you can define network names and their corresponding model files (in .pb format). Currently, only the converted AlexNet model is officially supported. You can try it with different models but might have to adjust code in `deep-spectrum/deep_spectrum_extractor/backend/extractor.py` in order to correctly identify your model's layer outputs in the graph's tensors. We plan on including a extraction backend for the new TensorFlow Hub (https://www.tensorflow.org/hub/) in the near future to make using different models easier.

 # Using the tool
 If you have installed the tool with pipenv, you can run it in different ways. Calling
 ```bash
 pipenv run extract_ds_features -h
```
from inside the `deep-spectrum` directory will run the tool from the virtualenv pipenv created for you automatically. You can also run
```bash
pipenv shell
extract_ds_features -h
```
from the same place. This will start a new shell for you in which the virtualenv is activated. For the following examples, we assume you used the second method.


Available command line options (also shown with `extract_ds_features -h`):


## Required options
| Option   | Description | Default |
|----------|-------------|---------|
| **-f**   | Specify the directory containing your *.wav* files here | None |
| **-o** | The location of the output feature file. Supported output formats are: Comma separated value files and arff files. If the specified output file's extension is *.arff*, arff is chosen as format, otherwise the output will be in comma separated value format. | None |


## Extracting features from audio chunks
| Option   | Description | Default |
|----------|-------------|---------|
| -t | Define window and hopsize for feature extraction. E.g `-t 1 0.5` extracts features from 1 second chunks every 0.5 seconds. | Extract from the whole audio file. |
| -start | Set a start time (in seconds) from which features should be extracted from the audio files. | 0 |
| -end | Set an end time until which features should be extracted from the audio files. | None |

## Setting parameters for the audio plots
| Option   | Description | Default |
|----------|-------------|---------|
| -mode | Type of plot to use in the system (Choose from: 'spectrogram', 'mel', 'chroma'). | spectrogram |
| -scale | Scale for the y-axis of the plots used by the system (Choose from: 'linear', 'log' and 'mel'). This is ignored if mode=chroma. (default: linear)
| -ylim | Specify a limit for the y-axis in the spectrogram plot in frequency. | None |
| -delta | If specified, derivatives of the given order of the selected features are displayed in the plots used by the system. | None |
| -nmel | Number of melbands used for computing the melspectrogram. Only takes effect with mode=mel. | 128 |
| -nfft | The length of the FFT window used for creating the spectrograms in number of samples. Consider choosing smaller values when extracting from small segments. | The next power of two from 0.025 x sampling_rate_of_wav |
| -cmap | Choose a matplotlib colourmap for creating the spectrogram plots. | viridis |

## Parameters for the feature extractor CNN
| Option   | Description | Default |
|----------|-------------|---------|
| -net | Choose the net for feature extraction as specified in the config file | alexnet |
| -layer | Name of the layer from which features should be extracted as specified in your caffe .prototxt file. | fc7 |

## Defining label information
| Option   | Description | Default |
|----------|-------------|---------|
| -l | Specify a comma separated values file containing labels for each *.wav* file. It has to include a header and the first column must specify the name of the audio file (with extension!) | None |
| --tc | Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column. | False |
| -el | Specify a single label that will be used for every input file explicitly. | None |
| --no_timestamps | Remove timestamps from the output. | Write timestamps in feature file. |
| --no_labels | Remove labels from the output. | Write labels in feature file. |

## Additional output 
| Option   | Description | Default |
|----------|-------------|---------|
| -specout | Specify a folder to save the plots used during extraction as .pngs | None |
| -wavout | Convenience function to write the chunks of audio data used in the extraction to the specified folder. | None |

## Configuration and Help
| Option   | Description | Default |
|----------|-------------|---------|
| -np | Specify the number of processes used for the extraction. Defaults to the number of available CPU cores | None |
| -config | The path to the configuration file used by the program can be given here. If the file does not exist yet, it is created and filled with standard settings. | deep.conf |
| -h | Show help. | None |
