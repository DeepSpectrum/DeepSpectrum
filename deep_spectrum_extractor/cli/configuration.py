import argparse
import configparser
import fnmatch
import re
from decimal import *
from itertools import chain
from os import listdir, makedirs, walk
from os.path import abspath, join, isfile, basename, expanduser, dirname, isdir, realpath

from deep_spectrum_extractor.tools.label_parser import LabelParser

getcontext().prec = 6


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
        self.gpu_mode = True
        self.device_ids = [0]
        self.number_of_processes = None
        self.folders = []
        self.output = ''
        self.cmap = 'viridis'
        self.label_file = None
        self.continuous_labels = False
        self.labels = None
        self.label_dict = {}
        self.layer = 'fc7'
        self.chunksize = None
        self.step = None
        self.start = 0
        self.nfft = 256
        self.y_limit = None
        self.reduced = None
        self.size = 227
        self.files = []
        self.output_spectrograms = None
        self.net = None
        self.transformer = None
        self.parser = None
        self.net = None
        self.config = None

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
        self.parser.add_argument('-l',
                                 help='csv file with the labels for the wavs in the form: \'test_001.wav, label\'. If nothing is specified here or under -labels, the name(s) of the directory/directories are used as labels.',
                                 default=None)
        self.parser.add_argument('-tc',
                                 help='Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column.',
                                 nargs='?', default=False, const=True)
        self.parser.add_argument('-el', nargs='+',
                                 help='Define labels for folders explicitly in format: labelForFirstFolder labelForSecondFolder ...',
                                 default=None)
        self.parser.add_argument('-cmap', default='viridis',
                                 help='define the matplotlib colour map to use for the spectrograms')
        self.parser.add_argument('-config',
                                 help='path to configuration file which specifies caffe model and weight files. If this file does not exist a new one is created and filled with the standard settings.',
                                 default=join(dirname(realpath(__file__)), 'deep.conf'))
        self.parser.add_argument('-layer', default='fc7',
                                 help='name of CNN layer (as defined in caffe prototxt) from which to extract the features.')
        # self.parser.add_argument('-chunksize', default=None, type=int,
        #                         help='define a chunksize in ms. wav data is split into chunks of this length before feature extraction.')
        # self.parser.add_argument('-step', default=None, type=int,
        #                         help='stepsize for creating the wav segments in ms. Defaults to the size of the chunks if -chunksize is given but -step is omitted.')
        self.parser.add_argument('-start',
                                 help='Set a start time from which features should be exrtracted from the audio files.',
                                 type=Decimal, default=0)
        self.parser.add_argument('-t',
                                 help='Extract deep spectrum features from windows with specified length and hopsize in seconds.',
                                 nargs=2, type=Decimal, default=[None, None])
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
        self.parser.add_argument('-net',
                                 help='specify the CNN that will be used for the feature extraction. This should be a key for which a model directory is assigned in the config file.',
                                 default='alexnet')

        args = vars(self.parser.parse_args())
        self.folders = args['f']
        self.cmap = args['cmap']
        self.output = abspath(args['o'])
        makedirs(dirname(self.output), exist_ok=True)
        self.label_file = args['l']
        self.labels = args['el']
        self.layer = args['layer']
        self.number_of_processes = args['np']

        # if either chunksize or step are not given they default to the value of the other given parameter
        # self.chunksize = args['chunksize'] if args['chunksize'] else args['step']
        # self.step = args['step'] if args['step'] else self.chunksize

        self.chunksize = args['t'][0]
        self.step = args['t'][1]
        self.start = args['start']
        self.continuous_labels = self.chunksize and args['tc'] and self.label_file
        self.nfft = args['nfft']
        self.reduced = args['reduced']
        self.output_spectrograms = abspath(args['specout']) if args['specout'] else None
        self.y_limit = args['ylim']
        self.net = args['net']
        self.config = args['config']

        # list all .wavs for the extraction found in the given folders
        self.files = list(chain.from_iterable([self._find_wav_files(folder) for folder in self.folders]))
        if not self.files:
            self.parser.error('No .wavs were found. Check the specified input paths.')

        if self.output_spectrograms:
            makedirs(self.output_spectrograms, exist_ok=True)

        if self.labels is not None and len(self.folders) != len(self.labels):
            self.parser.error(
                'Labels have to be specified for each folder: ' + str(len(self.folders)) + ' expected, ' + str(
                    len(self.labels)) + ' received.')
        print('Parsing labels...')
        if self.label_file is None:
            self._create_labels_from_folder_structure()
        else:
            self._read_label_file()

        self._load_config()
        self._configure_caffe()

    @staticmethod
    def _find_wav_files(folder):
        globexpression = '*.wav'
        reg_expr = re.compile(fnmatch.translate(globexpression), re.IGNORECASE)
        wavs = []
        for root, dirs, files in walk(folder, topdown=True):
            wavs += [join(root, j) for j in files if re.match(reg_expr, j)]
        return wavs

    def _read_label_file(self):
        """
        Read labels from either .csv or .tsv files
        :param parser: commandline parser
        :return: Nothing
        """

        # delimiters are decided by the extension of the labels file
        if self.label_file.endswith('.tsv'):
            parser = LabelParser(self.label_file, delimiter='\t', timecontinuous=self.continuous_labels)
        else:
            parser = LabelParser(self.label_file, delimiter=',', timecontinuous=self.continuous_labels)

        parser.parse_labels()
        self.label_dict = parser.label_dict
        self.labels = parser.labels

        file_names = set(map(basename, self.files))

        # check if labels are missing for specific files
        missing_labels = file_names.difference(self.label_dict)
        if missing_labels:
            self.parser.error('No labels for: ' + ', '.join(missing_labels))

    @staticmethod
    def _is_number(s):
        try:
            complex(s)  # for int, long, float and complex
        except ValueError:
            return False

        return True

    def _create_labels_from_folder_structure(self):
        """
        If no label file is given, either explicit labels or the folder structure is used as class values for the input.
        :return: Nothing
        """
        if self.labels is None:
            self.label_dict = {basename(wav): [basename(dirname(wav))] for wav in self.files}
        else:
            # map the labels given on the commandline to all files in a given folder in the order both appear in the
            # parsed options.
            self.label_dict = {basename(wav): [self.labels[folder_index]] for folder_index, folder in
                               enumerate(self.folders) for
                               wav in
                               self._find_wav_files(folder)}
        labels = list(map(lambda x: x[0], self.label_dict.values()))
        self.labels = [('class', set(labels))]

    def _load_config(self):
        """
        Parses the configuration file given on the commandline. If it does not exist yet, creates a new one containing
        standard settings.
        :param conf_file: configuration file to parse or create
        :return: Nothing
        """
        conf_parser = configparser.ConfigParser()

        # check if the file exists and parse it
        if isfile(self.config):
            print('Found config file ' + self.config)
            conf_parser.read(self.config)
            main_conf = conf_parser['main']
            self.gpu_mode = int(main_conf['gpu']) == 1
            self.device_ids = list(map(int, main_conf['device_ids'].split(',')))
            self.size = int(main_conf['size'])

            net_conf = conf_parser['nets']
            if self.net in net_conf:
                self.model_directory = net_conf[self.net]
            else:
                self.parser.error('No model directory defined for {} in {}'.format(self.net, self.config))


        # if not, create it with standard settings
        else:
            print('Writing standard config to ' + self.config)
            main_conf = {'gpu': '1' if self.gpu_mode else '0',
                         'device_ids': str(','.join(map(str, self.device_ids))),
                         'size': str(self.size)}
            net_conf = {'alexnet': self.model_directory}
            conf_parser['main'] = main_conf
            conf_parser['nets'] = net_conf
            with open(self.config, 'w') as configfile:
                conf_parser.write(configfile)

    def _configure_caffe(self):
        """
        Sets up the pre-trained CNN used for extraction.
        :param parser: commandline parser object used in the set up
        :return: Nothing
        """
        directory = self.model_directory

        if not isdir(self.model_directory):
            self.parser.error(
                'Directory {} specified in {} for net {} does not exist!'.format(self.model_directory, self.config,
                                                                                 self.net))
        # load model definition
        model_defs = [join(directory, file) for file in listdir(directory) if file.endswith('deploy.prototxt')]
        if model_defs:
            self.model_def = model_defs[0]
            print('CaffeNet definition: ' + self.model_def)
        else:
            self.model_def = ''
            self.parser.error('No model definition found in ' + directory + '.')

        # load model wights
        possible_weights = [join(directory, file) for file in listdir(directory)
                            if file.endswith('.caffemodel')]
        if possible_weights:
            self.model_weights = possible_weights[0]
            print('CaffeNet weights: ' + self.model_weights)
        else:
            self.parser.error('No model weights found in ' + directory + '.')
