import argparse
import configparser
import fnmatch
import re
from decimal import *
from os import listdir, makedirs, walk

from matplotlib import cm
from os.path import abspath, join, isfile, basename, expanduser, dirname, isdir, realpath

import deep_spectrum_extractor.tf_models as tf_models
from deep_spectrum_extractor.backend.plotting import PLOTTING_FUNCTIONS
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
        self.model_weights = join(expanduser('~'), 'tf_models/bvlc_alexnet.npy')
        self.number_of_processes = None
        self.input = []
        self.label_file = None
        self.reduced = None
        self.files = []
        self.output_spectrograms = None
        self.net = None
        self.parser = None
        self.config = None
        self.plotting_args = {}
        self.extraction_args = {}
        self.writer_args = {}
        self.backend = 'caffe'

    def parse_arguments(self):
        """
        Creates a commandline parser and handles the given options.
        :return: Nothing
        """
        self.parser = argparse.ArgumentParser(description='Extract deep spectrum features from wav files',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        required_named = self.parser.add_argument_group('Required named arguments')
        required_named.add_argument('-f', help='folder(s) where your wavs reside', required=True)
        required_named.add_argument('-o',
                                    help='the file which the features are written to. Supports csv and arff formats',
                                    required=True)
        self.parser.add_argument('-l',
                                 help='csv file with the labels for the wavs in the form: \'test_001.wav, label\'. If nothing is specified here or under -labels, the name(s) of the directory/directories are used as labels.',
                                 default=None)
        self.parser.add_argument('--tc',
                                 help='Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column.',
                                 action='store_true')
        self.parser.add_argument('-el', nargs=1,
                                 help='Define an explicit label for the input wavs.',
                                 default=None)
        self.parser.add_argument('-cmap', default='viridis',
                                 help='define the matplotlib colour map to use for the spectrograms',
                                 choices=sorted([m for m in cm.cmap_d]))
        self.parser.add_argument('-config',
                                 help='path to configuration file which specifies caffe model and weight files. If this file does not exist a new one is created and filled with the standard settings.',
                                 default=join(dirname(realpath(__file__)), 'deep.conf'))
        self.parser.add_argument('-layer', default='fc7',
                                 help='name of CNN layer (as defined in caffe prototxt) from which to extract the features.')
        self.parser.add_argument('-start',
                                 help='Set a start time from which features should be extracted from the audio files.',
                                 type=Decimal, default=0)
        self.parser.add_argument('-end',
                                 help='Set a end time until which features should be extracted from the audio files.',
                                 type=Decimal, default=None)
        self.parser.add_argument('-t',
                                 help='Extract deep spectrum features from windows with specified length and hopsize in seconds.',
                                 nargs=2, type=Decimal, default=[None, None])
        self.parser.add_argument('-nfft', default=None,
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
        self.parser.add_argument('-wavout',
                                 help='Convenience function to write the chunks of audio data used in the extraction to the specified folder.',
                                 default=None)

        self.parser.add_argument('-net',
                                 help='specify the CNN that will be used for the feature extraction. You need to specify a valid weight file in .npy format in your configuration file for this network.',
                                 default='AlexNet')

        self.parser.add_argument('-mode', help='Type of plot to use in the system.', default='spectrogram',
                                 choices=PLOTTING_FUNCTIONS.keys())
        self.parser.add_argument('-scale',
                                 help='Scale for the y-axis of the plots used by the system. Defaults to \'chroma\' in chroma mode.',
                                 default='linear',
                                 choices=['linear', 'log', 'mel'])
        self.parser.add_argument('-delta', type=_check_positive,
                                 help='If given, derivatives of the given order of the selected features are displayed in the plots used by the system.',
                                 default=None)
        self.parser.add_argument('--no_timestamps', action='store_true',
                                 help='Remove timestamps from the output.')
        self.parser.add_argument('-nmel', type=int,
                                 help='Number of melbands used for computing the melspectrogram.',
                                 default=128)
        self.parser.add_argument('-batch_size', type=int,
                                 help='Maximum batch size for feature extraction. Adjust according to your gpu memory size.',
                                 default=128)
        self.parser.add_argument('--no_labels', action='store_true',
                                 help='Do not write class labels to the output.')

        args = vars(self.parser.parse_args())
        self.input = args['f']
        self.net = args['net']
        self.config = args['config']
        self.reduced = args['reduced']
        self.output_spectrograms = abspath(args['specout']) if args['specout'] else None
        self.output_wavs = abspath(args['wavout']) if args['wavout'] else None
        self.label_file = args['l']

        self.number_of_processes = args['np']

        # arguments for the plotting functions
        self.plotting_args['cmap'] = args['cmap']
        self.plotting_args['mode'] = args['mode']
        self.plotting_args['scale'] = args['scale']
        self.plotting_args['delta'] = args['delta']
        self.plotting_args['cmap'] = args['cmap']
        self.plotting_args['ylim'] = args['ylim']
        self.plotting_args['nfft'] = args['nfft']
        self.plotting_args['start'] = args['start']
        self.plotting_args['end'] = args['end']
        self.plotting_args['window'] = args['t'][0]
        self.plotting_args['hop'] = args['t'][1]
        if self.plotting_args['mode'] == 'mel':
            self.plotting_args['melbands'] = args['nmel']
        if self.plotting_args['mode'] == 'chroma':
            self.plotting_args['scale'] = 'chroma'

        # arguments for extraction functions
        self.extraction_args['layer'] = args['layer']
        self.extraction_args['batch_size'] = args['batch_size']

        # arguments for writer
        self.writer_args['output'] = abspath(args['o'])
        makedirs(dirname(self.writer_args['output']), exist_ok=True)
        self.writer_args['window'] = args['t'][0]
        self.writer_args['hop'] = args['t'][1]
        self.writer_args['start'] = args['start']
        self.writer_args['continuous_labels'] = ('window' in self.plotting_args) and args['tc'] and self.label_file
        self.writer_args['labels'] = args['el']
        self.writer_args['no_timestamps'] = args['no_timestamps']
        self.writer_args['no_labels'] = args['no_labels']

        # list all .wavs for the extraction found in the given folders
        self.files = self._find_wav_files(self.input)
        if not self.files:
            self.parser.error('No .wavs were found. Check the specified input path.')

        if self.output_spectrograms:
            makedirs(self.output_spectrograms, exist_ok=True)

        # if self.writer_args['labels'] is not None and len(self.input) != len(self.writer_args['labels']):
        #     self.parser.error(
        #         'Labels have to be specified for each folder: ' + str(len(self.input)) + ' expected, ' + str(
        #             len(self.writer_args['labels'])) + ' received.')
        print('Parsing labels...')
        if self.label_file is None:
            self._create_labels_from_folder_structure()
        else:
            self._read_label_file()

        self._load_config()

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
            parser = LabelParser(self.label_file, delimiter='\t', timecontinuous=self.writer_args['continuous_labels'])
        else:
            parser = LabelParser(self.label_file, delimiter=',', timecontinuous=self.writer_args['continuous_labels'])

        parser.parse_labels()

        self.writer_args['label_dict'] = parser.label_dict
        self.writer_args['labels'] = parser.labels

        file_names = set(map(basename, self.files))

        # check if labels are missing for specific files
        missing_labels = file_names.difference(self.writer_args['label_dict'])
        if missing_labels:
            self.parser.error('No labels for: ' + ', '.join(missing_labels))

    def _create_labels_from_folder_structure(self):
        """
        If no label file is given, either explicit labels or the folder structure is used as class values for the input.
        :return: Nothing
        """
        if self.writer_args['labels'] is None:
            self.writer_args['label_dict'] = {basename(wav): [basename(dirname(wav))] for wav in self.files}
        else:
            # map the labels given on the commandline to all files in a given folder in the order both appear in the
            # parsed options.
            self.writer_args['label_dict'] = {basename(wav): self.writer_args['labels'] for wav in
                                              self.files}
        labels = sorted(list(map(lambda x: x[0], self.writer_args['label_dict'].values())))
        self.writer_args['labels'] = [('class', set(labels))]

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
            self.plotting_args['size'] = int(main_conf['size'])
            self.backend = main_conf['backend']
            self.extraction_args['gpu'] = int(main_conf['gpu']) == 1

            if self.backend == 'caffe':
                print('Using caffe backend as specified in {}'.format(self.config))
                try:
                    import caffe
                except ImportError:
                    self.parser.error('No caffe installation found!')
                net_conf = conf_parser['caffe-nets']
                if self.net in net_conf:
                    self.extraction_args['def_path'], self.extraction_args['weights_path'] = _find_caffe_files(
                        net_conf[self.net])
                    if not self.extraction_args['def_path']:
                        self.parser.error('No model definition for {} found in {}'.format(self.net, net_conf[self.net]))
                    if not self.extraction_args['weights_path']:
                        self.parser.error('No model weights for {} found in {}'.format(self.net, net_conf[self.net]))
                else:
                    self.parser.error('No model path defined for {} in {}'.format(self.net, self.config))

            elif self.backend == 'tensorflow':
                print('Using tensorflow backend as specified in {}'.format(self.config))
                try:
                    import tensorflow
                except ImportError:
                    self.parser.error('No tensorflow installation found!')
                net_conf = conf_parser['tensorflow-nets']
                if self.net in [model.__name__ for model in tf_models.get_models()]:
                    self.extraction_args['net_name'] = self.net
                    if self.net in net_conf:
                        self.extraction_args['weights_path'] = net_conf[self.net]
                    else:
                        self.parser.error('No model weights defined for {} in {}'.format(self.net, self.config))
                else:
                    self.parser.error('No model definition exists for {}. Available tf_models: {}'.format(self.net,
                                                                                                          [
                                                                                                              model.__name__
                                                                                                              for model
                                                                                                              in
                                                                                                              tf_models.get_models()]))
            else:
                self.parser.error(
                    'Unknown backend \'{}\' defined in {}. Available backends: tensorflow, caffe'.format(self.backend,
                                                                                                         self.config))

        # if not, create it with standard settings
        else:
            print('Writing standard config to ' + self.config)
            main_conf = {'size': str(227), 'gpu': str(1), 'backend': 'caffe'}
            tensorflow_net_conf = {model.__name__: '# Path to model weights (.npy) go here.' for model in
                                   tf_models.get_models()}
            caffe_net_conf = {
                model.__name__: '# Path to model folder containing model definition (.prototxt) and weights (.caffemodel) go here.'
                for model in
                tf_models.get_models()}
            conf_parser['main'] = main_conf
            conf_parser['tensorflow-nets'] = tensorflow_net_conf
            conf_parser['caffe-nets'] = caffe_net_conf
            with open(self.config, 'w') as configfile:
                conf_parser.write(configfile)
                self.parser.error('Please initialize your configuration file in {}'.format(self.config))


def _find_caffe_files(directory):
    if not isdir(directory):
        return None, None
    # load model definition
    model_defs = [join(directory, file) for file in listdir(directory) if file.endswith('deploy.prototxt')]
    if model_defs:
        model_def = model_defs[0]
        print('CaffeNet definition: ' + model_def)
    else:
        model_def = None

    # load model wights
    possible_weights = [join(directory, file) for file in listdir(directory)
                        if file.endswith('.caffemodel')]
    if possible_weights:
        model_weights = possible_weights[0]
        print('CaffeNet weights: ' + model_weights)
    else:
        model_weights = None

    return model_def, model_weights


def _check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue
