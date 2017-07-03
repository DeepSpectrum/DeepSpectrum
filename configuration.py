import argparse
import configparser
import csv
from itertools import chain
from os import listdir
from os.path import join, isfile, basename, expanduser, dirname, isdir


class Configuration:
    """
    This class handles the configuration of the deep spectrum extractor by reading commandline options and the
    configuration file. It then parses the labels for the audio files and configures the Caffe Network used for
    extraction.
    """

    def __init__(self):
        # set default values
        self.model_directory = join(expanduser('~'), 'caffe-master/models/bvlc_alexnet')
        self.model_def = ''
        self.model_weights = ''
        self.sr_directory = None
        self.sr_model_def = ''
        self.sr_model_weights = ''
        self.gpu_mode = True
        self.device_ids = []
        self.number_of_processes = None
        self.folders = []
        self.output = ''
        self.cmap = 'viridis'
        self.label_file = None
        self.labels = None
        self.label_dict = {}
        self.layer = 'fc7'
        self.chunksize = None
        self.step = None
        self.nfft = 256
        self.y_limit = None
        self.reduced = None
        self.size = 227
        self.files = []
        self.output_spectrograms = None
        self.net = None
        self.transformer = None
        self.parser = None
        self.scale = None

    def parse_arguments(self):
        """
        Creates a commandline parser and handles the given options.
        :return: Nothing
        """
        self.parser = argparse.ArgumentParser(description='Extract deep spectrum features from wav files',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        required_named = self.parser.add_argument_group('Required named arguments')
        required_named.add_argument('-f', nargs='+', help='folder(s) where your wavs reside', required=True)
        required_named.add_argument('-o',
                                    help='the file which the features are written to. Supports csv and arff formats',
                                    required=True)
        self.parser.add_argument('-lf',
                                 help='csv file with the labels for the wavs in the form: \'test_001.wav, label\'. If nothing is specified here or under -labels, the name(s) of the directory/directories are used as labels.',
                                 default=None)
        self.parser.add_argument('-labels', nargs='+',
                                 help='define labels for folders explicitly in format: labelForFirstFolder labelForSecondFolder ...',
                                 default=None)
        self.parser.add_argument('-cmap', default='viridis',
                                 help='define the matplotlib colour map to use for the spectrograms')
        self.parser.add_argument('-config',
                                 help='path to configuration file which specifies caffe model and weight files. If this file does not exist a new one is created and filled with the standard settings-',
                                 default="deep.conf")
        self.parser.add_argument('-layer', default='fc7',
                                 help='name of CNN layer (as defined in caffe prototxt) from which to extract the features.')
        self.parser.add_argument('-chunksize', default=None, type=int,
                                 help='define a chunksize in ms. wav data is split into chunks of this length before feature extraction.')
        self.parser.add_argument('-step', default=None, type=int,
                                 help='stepsize for creating the wav segments in ms. Defaults to the size of the chunks if -chunksize is given but -step is omitted.')
        self.parser.add_argument('-nfft', default=256,
                                 help='specify the size for the FFT window in number of samples', type=int)
        self.parser.add_argument('-reduced', nargs='?',
                                 help='a reduced version of the feature set is written to the given location.',
                                 default=None, const='deep_spectrum_reduced.arff')
        self.parser.add_argument('-np', type=int,
                                 help='define the number of processes used in parallel for the extraction. If None defaults to cpu-count',
                                 default=None)
        self.parser.add_argument('-ylim', type=int,
                                 help='define a limit for the y-axis for plotting the spectrograms',
                                 default=None)

        self.parser.add_argument('-specout',
                                 help='define an existing folder where spectrogram plots should be saved during feature extraction. By default, spectrograms are not saved on disk to speed up extraction.',
                                 default=None)
        self.parser.add_argument('-sr',
                                 help='define a factor for super resolution scaling using VDSR. Warning: Very memory intensive.',
                                 default=None, type=int, nargs='?', const=2)

        args = vars(self.parser.parse_args())
        self.folders = args['f']
        self.cmap = args['cmap']
        self.output = args['o']
        self.label_file = args['lf']
        self.labels = args['labels']
        self.layer = args['layer']
        self.number_of_processes = args['np']
        self.scale = args['sr']

        # if either chunksize or step are not given they default to the value of the other given parameter
        self.chunksize = args['chunksize'] if args['chunksize'] else args['step']
        self.step = args['step'] if args['step'] else self.chunksize
        self.nfft = args['nfft']
        self.reduced = args['reduced']
        self.output_spectrograms = args['specout']
        self.y_limit = args['ylim']

        # list all .wavs for the extraction found in the given folders
        self.files = list(chain.from_iterable([self._find_wav_files(folder) for folder in self.folders]))
        if not self.files:
            self.parser.error('No .wavs were found. Check the specified input paths.')

        if self.output_spectrograms and not isdir(self.output_spectrograms):
            self.parser.error('Spectrogram directory \'' + self.output_spectrograms + '\' does not exist.')

        if self.labels is not None and len(self.folders) != len(self.labels):
            self.parser.error(
                'Labels have to be specified for each folder: ' + str(len(self.folders)) + ' expected, ' + str(
                    len(self.labels)) + ' received.')
        print('Parsing labels...')
        if self.label_file is None:
            self._create_labels_from_folder_structure()
        else:
            self._read_label_file()

        self._load_config(args['config'])
        self._configure_caffe()

    @staticmethod
    def _find_wav_files(folder):
        if listdir(folder):
            wavs = [join(folder, wav_file) for wav_file in listdir(folder) if
                    isfile(join(folder, wav_file)) and (wav_file.endswith('.wav') or wav_file.endswith('.WAV'))]
            return wavs + list(chain.from_iterable(
                [Configuration._find_wav_files(join(folder, subfolder)) for subfolder in listdir(folder) if
                 isdir(join(folder, subfolder))]))
        else:
            return []

    def _read_label_file(self):
        """
        Read labels from either .csv or .tsv files
        :param parser: commandline parser
        :return: Nothing
        """

        # delimiters are decided by the extension of the labels file
        if self.label_file.endswith('.tsv'):
            reader = csv.reader(open(self.label_file), delimiter="\t")
        else:
            reader = csv.reader(open(self.label_file))
        self.label_dict = {}

        # a list of distinct labels is needed for deciding on the nominal class values for .arff files
        self.labels = set([])

        # parse the label file line by line
        for row in reader:
            key = row[0]
            self.label_dict[key] = row[1]
            self.labels.add(row[1])
        file_names = set(map(basename, self.files))

        # check if labels are missing for specific files
        missing_labels = file_names.difference(self.label_dict)
        if missing_labels:
            self.parser.error('No labels for: ' + ', '.join(missing_labels))

    def _create_labels_from_folder_structure(self):
        """
        If no label file is given, either explicit labels or the folder structure is used as class values for the input.
        :return: Nothing
        """
        if self.labels is None:
            self.label_dict = {basename(wav): basename(dirname(wav)) for wav in self.files}
        else:
            # map the labels given on the commandline to all files in a given folder in the order both appear in the
            # parsed options.
            self.label_dict = {basename(wav): self.labels[folder_index] for folder_index, folder in enumerate(self.folders) for
                               wav in
                               self._find_wav_files(folder)}

    def _load_config(self, conf_file):
        """
        Parses the configuration file given on the commandline. If it does not exist yet, creates a new one containing
        standard settings.
        :param conf_file: configuration file to parse or create
        :return: Nothing
        """
        conf_parser = configparser.ConfigParser()

        # check if the file exists and parse it
        if isfile(conf_file):
            print('Found config file ' + conf_file)
            conf_parser.read(conf_file)
            conf = conf_parser['main']
            self.model_directory = conf['caffe_model_directory']
            self.sr_directory = conf['VDSR_directory']
            self.gpu_mode = int(conf['gpu']) == 1
            self.device_ids = list(map(int, conf['device_ids'].split(',')))
            self.size = int(conf['size'])

        # if not, create it with standard settings
        else:
            print('Writing standard config to ' + conf_file)
            conf = {'caffe_model_directory': self.model_directory,
                    'VDSR_directory': '?',
                    'gpu': '1' if self.gpu_mode else '0',
                    'device_ids': str(','.join(self.device_ids)),
                    'size': str(self.size)}
            conf_parser['main'] = conf
            with open(conf_file, 'w') as configfile:
                conf_parser.write(configfile)

    def _configure_caffe(self):
        """
        Sets up the pre-trained CNN used for extraction.
        :param parser: commandline parser object used in the set up
        :return: Nothing
        """
        directory = self.model_directory
        # load model definition
        model_defs = [join(directory, file) for file in listdir(directory) if file.endswith('deploy.prototxt')]
        if model_defs:
            self.model_def = model_defs[0]
            print('CaffeNet definition: ' + self.model_def)
        else:
            self.model_def = ''
            self.parser.error("No model definition found in " + directory + '.')

        # load model wights
        possible_weights = [join(directory, file) for file in listdir(directory)
                            if file.endswith('.caffemodel')]
        if possible_weights:
            self.model_weights = possible_weights[0]
            print('CaffeNet weights: ' + self.model_weights)
        else:
            self.parser.error("No model weights found in " + directory + '.')

        if self.scale:
            directory = self.sr_directory

            # load model definition
            model_defs = [join(directory, file) for file in listdir(directory) if file.endswith('deploy.prototxt')]
            if model_defs:
                self.sr_model_def = model_defs[0]
                print('CaffeNet definition: ' + self.sr_model_def)
            else:
                self.sr_model_def = ''
                self.parser.error("No model definition found in " + directory + '.')

            # load model wights
            possible_weights = [join(directory, file) for file in listdir(directory)
                                if file.endswith('.caffemodel')]
            if possible_weights:
                self.sr_model_weights = possible_weights[0]
                print('CaffeNet weights: ' + self.sr_model_weights)
            else:
                self.parser.error("No model weights found in " + directory + '.')

