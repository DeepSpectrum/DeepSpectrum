# Deep Spectrum Feature Extraction

## Dependencies
* Python 3.5+
* NumPy
* matplotlib
* imread
* scipy
* pysoundfile (on Linux you also need to install libsndfile manually, e.g., by `sudo apt-get install libsndfile1`)
* tqdm
* pandas
* liac-arff (imports as 'arff')
* librosa
* caffe with all dependencies. For installation instructions: 
  [https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide) or 
  [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html). 
  Make sure you can do `import caffe` without errors from the python prompt in your commandline 
  of choice. 
* alternatively, the tool can also be run with a tensorflow backend based on (https://github.com/ethereon/caffe-tensorflow). 

## Installation
For the caffe version, all dependencies apart from caffe itself can be installed by running
```pip install -e .``` from the project root directory.


For the tensorflow version, simply run 
```pip install -e .['tensorflow-gpu']```
or 
```pip install -e .['tensorflow']```
if you prefer a cpu only version. 

Installing into a python virtual environment is recommended in both cases (for caffe, the virtualenvironment has to be created with the `--system-site-packages` option).



## Configuration
Apart from the commandline options, the feature extractor is configured via a 
configuration file. This file specifies 4 things:

1. Whether you would like to use your GPU for computation (recommended).
2. The width and height of the spectrograms plotted during the feature extraction
   in pixels. Specify only one integer here, as the plots used by the system 
   have equal height and width. This parameter heavily impacts performance:
   if set to the input dimensions of your CNN (look at your *.prototxt for this)
   the speed of the feature extraction can be nearly doubled as the spectrogram
   plots don't have to be rescaled.
3. The backend you would like to use for the feature extraction. Caffe and tensorflow are available.
4. Depending on the backend:
    * Caffe: The directories of your caffe CNN models. These directories should contain the 
   model specification (a file that ends with *deploy.prototxt*) and weights 
   (a larger file that ends with *.caffemodel*) and can be specified after keys that are then accessible from the commandline.
    * Tensorflow: The paths to model weights (.npy) for the caffe-tensorflow models. These can be obtained by converting corresponding .caffemodel files with https://github.com/ethereon/caffe-tensorflow (no caffe installation is needed dfor this). 




```
[main]
size = 227
gpu = 1
backend = caffe

[tensorflow-nets]
vgg16 = # Path to model weights (.npy) go here.
resnet50 = # Path to model weights (.npy) go here.
nin = # Path to model weights (.npy) go here.
resnet101 = # Path to model weights (.npy) go here.
alexnet = # Path to model weights (.npy) go here.
googlenet = # Path to model weights (.npy) go here.
caffenet = # Path to model weights (.npy) go here.
resnet152 = # Path to model weights (.npy) go here.

[caffe-nets]
vgg16 = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
resnet50 = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
nin = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
resnet101 = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
alexnet = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
googlenet = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
caffenet = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
resnet152 = # Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.
```

The above configuration file is created upon first launching the system or specifying a nonexistent config file on the commandline.

## Usage
### Basics
Basic usage from the commandline requires two arguments: `-f` specifies the 
folder(s) containing the .wav files you want to extract the features from 
and `-o` is the filepath to which the output is written to. An example prompt 
for this basic scenario is described below:
* `python3 main.py -f /wavs/angry /wavs/sad -o angrySadDeepSpectrum.arff`

This will extract deep spectrum features by using the net, image size and device
(GPU or CPU) sepcified in the standard configuration file [deep.conf](deep.conf)
(cf. ). 
The extraction is performed by using the colourmap *viridis* for spectrogram 
plots of whole audio files, and layer *fc7*. 
As no label file or explicit labels have been given, the feature vectors for 
each .wav file are assigned the names of their parent directories as label, i.e. 
in the example *angry* and *sad*. The features are finally written to 
*angrySadDeepSpectrum.arff* in arff format.

### Advanced: Specifying labels
Labels for the *.wav* files can be given explicitly either by providing a label 
file or specifying labels for each folder of the extraction:
* label files (*.tsv*, *.csv* are supported) can be specified by `-l labels.csv`.
  They have to follow the format *file_name.wav* **delimiter** *label* where **delimiter**
  must be `,` for *.csv* and `\t` (tab) for *.tsv*
* labels for each folder can be specified by `-el labelForFolder1 labelForFolder2`.
  The number of specified labels must match the number of folders given after `-f`.

### Advanced: Extracting features from audio segments
Instead of using the whole audio file to create a single feature vector, 
features can also be extracted from equally sized chunks of the audio file
resulting in indexed time series features. This is controlled from the 
commandline by the parameter -t windowSize hopSize, where windowSize and hopSize
should be given in seconds.

### Commandline options
A detailed description of the commandline options is given below. Apart from 
`-f` and `-o` none of the options are required.<br>

### Image feature extractor
For convenience, a extractor for traditional CNN-descriptors is also included. 
The tool works on collections of .png images and can be used with the main 
tool's arguments - except for those specific to creating audio plots:
* `image_cnn_features -f ~/Desktop/spectrograms -o ~/Desktop/train.arff -net AlexNet -layer fc7 -l ~/Desktop/labels.csv`


| Option   | Description | Default |
|----------|-------------|---------|
| **-f**   | Specify the directory containing your *.wav* files here | None |
| **-o** | The location of the output feature file. Supported output formats are: Comma separated value files and arff files. If the specified output file's extension is *.arff*, arff is chosen as format, otherwise the output will be in comma separated value format. | None |
| -l | Specify a comma separated values file containing labels for each *.wav* file | None |
| -tc | Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column. | False |
| -el | Specify a label explicitly for the input files. | None |
| --no_timestamps | Remove timestamps from the output. | Write timestamps in feature file. |
| --no_labels | Remove labels from the output. | Write labels in feature file. |
| -mode | Type of plot to use in the system (Choose from: 'spectrogram', 'mel', 'chroma'). | spectrogram |
| -scale | Scale for the y-axis of the plots used by the system. Defaults to 'chroma' in chroma mode. (default: linear)
| -ylim | Specify a limit for the y-axis in the spectrogram plot in frequency. | None |
| -delta | If specified, derivatives of the given order of the selected features are displayed in the plots used by the system. | None |
| -nmel | Number of melbands used for computing the melspectrogram. | 128 |
| -nfft | The length of the FFT window used for creating the spectrograms in number of samples. Consider choosing smaller values when extracting from small segments. | The next power of two from 0.025*sampling_rate_of_wav |
| -cmap | Choose a matplotlib colourmap for creating the spectrogram plots. | viridis |
| -net | Choose the net for feature extraction as specified in the config file | alexnet |
| -layer | Name of the layer from which features should be extracted as specified in your caffe .prototxt file. | fc7 |
| -start | Set a start time (in seconds) from which features should be extracted from the audio files. | 0 |
| -end | Set a end time until which features should be extracted from the audio files. | None |
| -t | Define window and hopsize. | None |
| -specout | Specify a folder to save the plots used during extraction as .pngs | None |
| -wavout | Convenience function to write the chunks of audio data used in the extraction to the specified folder. | None |
| -reduced | If a filepath is given here, an additional reduced version of the output is computed after feature extraction and written to the path. The feature reduction simply removes attributes that have a value of zero for all instances. | None |
| -np | Specify the number of processes used for the extraction. Defaults to the number of available CPU cores | None |
| -config | The path to the configuration file used by the program can be given here. If the file does not exist yet, it is created and filled with standard settings. | deep.conf |
| -h | Show help. | None |

### Classifiers

The tool also includes classifiers to be used with the extracted features (csv or arff).

#### Linear SVM

If the package is installed, a linear SVM classifier can be called with: 

```linear_svm train.arff devel.arff test.arff```

per default, this evaluates SVM complexities on a logarithmic scale between 1 
and 10^-9 by training the classifier on the first feature file and evaluating it 
on the second one. Another evaluation is done by combining the first to partitions
and evaluating on the last. If only two partitions are given, the first evaluation 
is performed by 10-fold CV. Additional arguments (e.g. for writing the results, 
specifying the complexities by hand or plotting a confusion matrix) are described 
by calling `-h`.

#### Neural Networks

Two types of neural network models are available: A regular feed-forward network 
and a recurrent neural network usable with either gru or lstm cells. The 
networks can be run with `dnn` and `rnn` and support three operating modes: 
`train`, `eval` and `predict`. All commands have the following common structure: 

```dnn|rnn train|eval|predict [feature_files]```

`train` takes two feature files as input: training data and evaluation 
data, whereas `eval` and `predict` use a single input file. All cli interface share
the `--model_dir` argument which specifies where checkpoints should be written
during trainig and from where a trained model should be loaded for evaluation/prediction.
All additional parameters for each network and operating mode are accessible via `-h`.
