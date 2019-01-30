**DeepSpectrum** is a Python toolkit for feature extraction from audio data with pre-trained Image Convolutional Neural Networks (CNNs). It features an extraction pipeline which first creates visual representations for audio data - plots of spectrograms or chromagrams - and then feeds them to a pre-trained Image CNN. Activations of a specific layer then form the final feature vectors.

**(c) 2017-2018 Shahin Amiriparian, Maurice Gerczuk, Sandra Ottl, Björn Schuller: Universität Augsburg**
Published under GPLv3, see the LICENSE.md file for details.

Please direct any questions or requests to Shahin Amiriparian (shahin.amiriparian at tum.de) or Maurice Gercuk (gerczuk at fim.uni-passau.de).

# Citing
If you use DeepSpectrum or any code from DeepSpectrum in your research work, you are kindly asked to acknowledge the use of DeepSpectrum in your publications.
> S. Amiriparian, M. Gerczuk, S. Ottl, N. Cummins, M. Freitag, S. Pugachevskiy, A. Baird and B. Schuller. Snore Sound Classification using Image-Based Deep Spectrum Features. In Proceedings of INTERSPEECH (Vol. 17, pp. 2017-434)


# Installation
This program provides a setup.py script which declares all dependencies.

## Dependencies
* Python >=3.6

## Deep Spectrum tool
We recommend that you install the DeepSpectrum tool into a virtual environment. To do so first create a new virtualenvironment:
```bash
virtualenv -p python3 ds_virtualenv
```
If you have a recent installation of tensorflow (>=1.12) on your system, you can also create a virtualenvironment that incorporates you system python packages:
```bash
virtualenv -p python3 --system-site-packages ds_virtualenv
```
This creates a minimal python installation in the folder "ds_virtualenv". You can choose a different name instead of "ds_virtualenv" if you like, but the guide assumes this name.
You can then activate the virtualenv (Linux):
```bash
source ds_virtualenv/bin/activate
```
Or for windows:
```bash
.\ds_virtualenv\Scripts\activate.bat
```

## Configuration
If you used the included script to download the AlexNet model, the tool is already configured correctly for [usage](#using-the-tool). Otherwise, you have to adjust your configuration file. The default file can be found in `deep-spectrum/src/cli/deep.conf`:
```
[main]
size = 227
gpu = 1
backend = tensorflow

[tensorflow-nets]
alexnet = AlexNet.pb
```
Under `tensorflow-nets` you can define network names and their corresponding model files (in .pb format). Currently, only the converted AlexNet model is officially supported. You can try it with different models but might have to adjust code in `deep-spectrum/src/backend/extractor.py` in order to correctly identify your model's layer outputs in the graph's tensors. We plan on including a extraction backend for the new TensorFlow Hub (https://www.tensorflow.org/hub/) in the near future to make using different models easier.

# Using the tool
You can access the scripts provided by the tool from the virtualenvironment. The feature extraction component is provided by `ds-features`.

## Features for AVEC2018 CES
The command below extracts features from overlapping 1 second windows spaced with a hop size of 0.1 seconds (`-t 1 0.1`) of the the file `Train_DE_01.wav`. It plots mel spectrograms (`-mode mel`) and feeds them to a pre-trained AlexNet model (`-net alexnet`). The activations on the fc7 layer (`-layer fc7`) are finally written to `Train_DE_01.arff` as feature vectors in arff format. `--no_labels` suppresses writing any labels to the output file.
```bash
ds-features -i Train_DE_01.wav -t 1 0.1 --no_labels -net alexnet -layer fc7 -mode mel -o Train_DE_01.arff
```

## Commandline Options
All options can also be displayed using `ds-features -h`.
### Required options
| Option   | Description | Default |
|----------|-------------|---------|
| **-i**   | Specify the directory containing your *.wav* files or the path to a single *.wav* file. | None |
| **-o** | The location of the output feature file. Supported output formats are: Comma separated value files and arff files. If the specified output file's extension is *.arff*, arff is chosen as format, otherwise the output will be in comma separated value format. | None |


### Extracting features from audio chunks
| Option   | Description | Default |
|----------|-------------|---------|
| -t | Define window and hopsize for feature extraction. E.g `-t 1 0.5` extracts features from 1 second chunks every 0.5 seconds. | Extract from the whole audio file. |
| -start | Set a start time (in seconds) from which features should be extracted from the audio files. | 0 |
| -end | Set an end time until which features should be extracted from the audio files. | None |

### Setting parameters for the audio plots
| Option   | Description | Default |
|----------|-------------|---------|
| -mode | Type of plot to use in the system (Choose from: 'spectrogram', 'mel', 'chroma'). | spectrogram |
| -scale | Scale for the y-axis of the plots used by the system (Choose from: 'linear', 'log' and 'mel'). This is ignored if mode=chroma or mode=mel. (default: linear)
| -ylim | Specify a limit for the y-axis in the spectrogram plot in frequency. | None |
| -delta | If specified, derivatives of the given order of the selected features are displayed in the plots used by the system. | None |
| -nmel | Number of melbands used for computing the melspectrogram. Only takes effect with mode=mel. | 128 |
| -nfft | The length of the FFT window used for creating the spectrograms in number of samples. Consider choosing smaller values when extracting from small segments. | The next power of two from 0.025 x sampling_rate_of_wav |
| -cmap | Choose a matplotlib colourmap for creating the spectrogram plots. | viridis |

### Parameters for the feature extractor CNN
| Option   | Description | Default |
|----------|-------------|---------|
| -net | Choose the net for feature extraction as specified in the config file | alexnet |
| -layer | Name of the layer from which features should be extracted as specified in your caffe .prototxt file. | fc7 |

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
| -l | Specify a comma separated values file containing labels for each *.wav* file. It has to include a header and the first column must specify the name of the audio file (with extension!) | None |
| --tc | Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column. | False |
| -el | Specify a single label that will be used for every input file explicitly. | None |
| --no_timestamps | Remove timestamps from the output. | Write timestamps in feature file. |
| --no_labels | Remove labels from the output. | Write labels in feature file. |

### Additional output 
| Option   | Description | Default |
|----------|-------------|---------|
| -specout | Specify a folder to save the plots used during extraction as .pngs | None |
| -wavout | Convenience function to write the chunks of audio data used in the extraction to the specified folder. | None |

### Configuration and Help
| Option   | Description | Default |
|----------|-------------|---------|
| -np | Specify the number of processes used for the extraction. Defaults to the number of available CPU cores | None |
| -config | The path to the configuration file used by the program can be given here. If the file does not exist yet, it is created and filled with standard settings. | deep.conf |
| -h | Show help. | None |

# Toolkit
Apart from the main feature extraction functionality, the toolkit also includes various other commandline scripts. An overview can be shown on via:
```bash
ds-help
```
| Script | Description |
|-----------------------------------------|--------------------------------------------------|
| ds-features | Extract deep spectrum features from wav files.                   |
| ds-image-features | Extract CNN-descriptors from images.                      |
| ds-plot | Create plots from wav files.                                     |
| ds-reduce | Reduce a list of feature files by removing zero features.      |
| ds-scikit | Train and evaluate optimized scikit-learn models.             |
| ds-dnn | Interface for training and evaluating a Deep Neural Network.     |
| ds-rnn | Interface for training and evaluating a Recurrent Neural Network.|
| ds-cm | Create a pdf plot from the textual representation of a confusion matrix. |
| ds-results | Tool to inspect, load and export results. |

For each tool, detailed usage descriptions are available with: 'ds-[tool] --help'


## Neural Networks

Two types of neural network models are available: A regular feed-forward network 
and a recurrent neural network usable with either gru or lstm cells. The 
networks can be run with `ds-dnn` and `ds-rnn` and support three operating modes: 
`train`, `eval` and `predict`. All commands have the following common structure: 

```bash
ds-dnn|ds-rnn train|eval|predict [feature_files]
```

`train` takes two feature files as input: training data and evaluation 
data, whereas `eval` and `predict` use a single input file. All cli interfaces share
the `--model_dir` argument which specifies where checkpoints should be written
during trainig and from where a trained model should be loaded for evaluation/prediction.
All additional parameters for each network and operating mode are accessible via `-h`.
