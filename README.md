# Deep Spectrum Feature Extraction

## Dependencies
* Python 3.5+
* NumPy
* matplotlib
* imread
* scipy
* tqdm
* pandas
* liac-arff
* caffe with all dependencies. For installation instructions: 
  [https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide](https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide) or 
  [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html). 
  Make sure you can do `import caffe` from the python propt in your commandline 
  of choice. 

## Usage
### Basics
Basic usage from the commandline requires two arguments: `-f` specifies the 
folder(s) containing the .wav files you want to extract the features from 
and `-o` is the filepath to which the output is written to. An example prompt 
for this basic scenario is described below:
* `python3 extract_deep_spectrum.py -f /wavs/angry /wavs/sad -o angrySadDeepSpectrum.arff`

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
Labels for the .wav files can be giiven explicitly either by providing a label 
file or specifying labels for each folder of the extraction:
* label files (*.tsv*, *.csv* are supported) can be specified by `-lf labels.csv`.
  They have to follow the format *file_name.wav* **delimiter** *label* where **delimiter**
  must be `,` for *.csv* and `\t` (tab) for *.tsv*
* labels for each folder can be specified by `-labels labelForFolder1 labelForFolder2`.
  The number of specified labels must match the number of folders given after `f`.



