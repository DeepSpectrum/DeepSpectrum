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
* caffe with all dependencies. For installation instructions: 
  [https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide) or 
  [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html). 
  Make sure you can do `import caffe` without errors from the python prompt in your commandline 
  of choice. 

## Configuration
Apart from the commandline options, the feature extractor is configured via a 
configuration file. This file specifies 4 things:
1. The directory of your caffe CNN model. This directory should contain the 
   model specification (a file that ends with *deploy.prototxt*) and weights 
   (a larger file that ends with *.caffemodel*) 
2. The id of the device used for computation. Only relevant in a multi GPU system
   -> 0 should be fine
3. Whether you would like to use your GPU for computation (recommended).
4. The width and height of the spectrograms plotted during the feature extraction
   in pixels. Specify only one integer here, as the plots used by the system 
   have equal height and width. This parameter heavily impacts performance:
   if set to the input dimensions of your CNN (look at your *.prototxt for this)
   the speed of the feature extraction can be nearly doubled as the spectrogram
   plots don't have to be rescaled.



`[main]
caffe_model_directory = ~/caffe-master/models/bvlc_alexnet
device_id = 0
gpu = 1
size = 227`


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
* label files (*.tsv*, *.csv* are supported) can be specified by `-lf labels.csv`.
  They have to follow the format *file_name.wav* **delimiter** *label* where **delimiter**
  must be `,` for *.csv* and `\t` (tab) for *.tsv*
* labels for each folder can be specified by `-labels labelForFolder1 labelForFolder2`.
  The number of specified labels must match the number of folders given after `-f`.

### Advanced: Extracting features from audio segments
Instead of using the whole audio file to create a single feature vector, 
features can also be extracted from equally sized chunks of the audio file
resulting in indexed time series features. This is controlled from the 
commandline by the parameters `-chunksize` and `-step`:
* the stepsize for the chunking of the audio files can be specified in ms after 
`-step`
* the length of the audio chunks in ms is controlled by `-chunksize`. 
* if only `-step` is specified, the chunksize defaults to the stepsize

Additionally, when extracting from audio segments, `-nfft`should also be set 
accordingly as for small segments the default of 256 samples used for the FFT is
often too large. 

### Commandline options
A detailed description of the commandline options is given below. Apart from 
`-f` and `-o` none of the options are required.<br>



| Option   | Description | Default |
|----------|-------------|---------|
| **-f**   | Specify the directory/directories containing your *.wav* files here | None |
| **-o** | The location of the output feature file. Supported output formats are: Comma separated value files and arff files. If the specified output file's extension is *.arff*, arff is chosen as format, otherwise the output will be in comma separated value format. | None |
| -lf | Specify a comma separated values file containing labels for each *.wav* file | None |
| -labels | Specify labels explicitly for each folder given after -f. Number of given labels has to match the number of specified folders. If both this and -lf are not specified, each .wav is assigned the name of its parent directory as label. | None |
| -cmap | Choose a matplotlib colourmap for creating the spectrogram plots. | viridis |
| -layer | Name of the layer from which features should be extracted as specified in your caffe .prototxt file. Only layers with 1-D output are supported | fc7 |
| -step | Configure a stepsize for segmentation of the wavs in ms | None |
| -chunksize | Define the length of the segments. Defaults to *chunksize* if `-step` is given but chunksize is omissed. | None |
| -nfft | The length of the FFT window used for creating the spectrograms in number of samples. Consider choosing smaller values when extracting from small segments. | 256 |
| -reduced | If a filepath is given here, an additional reduced version of the output is computed after feature extraction and written to the path. The feature reduction simply removes attributes that have a value of zero for all instances. | None |
| -config | The path to the configuration file used by the program can be given here. If the file does not exist yet, it is created and filled with standard settings. | deep.conf |
| -specout | Specify an existing folder to save the spectrograms as .pngs | None |