import argparse
import configparser
import fnmatch
import re
from decimal import *
from matplotlib import cm
from os import makedirs, walk
from os.path import abspath, join, isfile, basename, expanduser, dirname, realpath

from deep_spectrum_extractor.backend.plotting import PLOTTING_FUNCTIONS
from deep_spectrum_extractor.tools.label_parser import LabelParser
from ..backend.extractor import TensorFlowExtractor

getcontext().prec = 6


class Configuration:
    """
    This class handles the configuration of the deep spectrum extractor by reading commandline options and the
    configuration file. It then also parses the labels for the audio files.
    """

    def __init__(self,
                 extraction=True,
                 plotting=True,
                 writer=True,
                 file_type='wav'):
        # set default values
        self.model_weights = 'AlexNet.pb'
        self.number_of_processes = None
        self.label_file = None
        self.files = []
        self.input = []
        self.file_type = file_type
        self.output_spectrograms = None
        self.net = None
        self.parser = None
        self.config = None
        self.plotting = plotting
        self.plotting_args = {}
        self.extraction = extraction
        self.extraction_args = {}
        self.writer = writer
        self.writer_args = {}
        self.backend = 'tensorflow'
        self.parsers = [self.general_parser(), self.label_parser()]
        if self.plotting:
            self.parsers.append(self.plotting_parser())
        if self.extraction:
            self.parsers.append(self.extraction_parser())
        if self.writer:
            self.parsers.append(self.writer_parser())

    def general_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-i', help='folder where your wav files reside. Alternatively, path to a single .wav file.', required=True)
        parser.add_argument(
            '-config',
            help=
            'path to configuration file which specifies caffe model and weight files. If this file does not exist a new one is created and filled with the standard settings.',
            default=join(dirname(realpath(__file__)), 'deep.conf'))
        parser.add_argument(
            '-np',
            type=int,
            help=
            'define the number of processes used in parallel for the extraction. If None defaults to cpu-count',
            default=None)
        parser.add_argument(
            '--tc',
            help=
            'Set labeling of features to time continuous mode. Only works in conjunction with -t and the specified label file has to provide labels for the specified hops in its second column.',
            action='store_true')
        return parser

    def plotting_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-start',
            help=
            'Set a start time from which features should be extracted from the audio files.',
            type=Decimal,
            default=0)
        parser.add_argument(
            '-end',
            help=
            'Set a stop time until which features should be extracted from the audio files.',
            type=Decimal,
            default=None)
        parser.add_argument(
            '-t',
            help=
            'Extract deep spectrum features from windows with specified length and hopsize in seconds.',
            nargs=2,
            type=Decimal,
            default=[None, None])
        parser.add_argument(
            '-nfft',
            default=None,
            help='specify the size for the FFT window in number of samples',
            type=int)
        parser.add_argument(
            '-cmap',
            default='viridis',
            help='define the matplotlib colour map to use for the spectrograms',
            choices=sorted([m for m in cm.cmap_d]))
        parser.add_argument(
            '-ylim',
            type=int,
            help='define a frequency limit (hz) for the y-axis for plotting the spectrograms',
            default=None)

        parser.add_argument(
            '-specout',
            help=
            'define an existing folder where spectrogram plots should be saved during feature extraction. By default, spectrograms are not saved on disk to speed up extraction.',
            default=None)
        parser.add_argument(
            '-wavout',
            help=
            'Convenience function to write the chunks of audio data used in the extraction to the specified folder.',
            default=None)
        parser.add_argument(
            '-mode',
            help='Type of plot to use in the system.',
            default='spectrogram',
            choices=PLOTTING_FUNCTIONS.keys())
        parser.add_argument(
            '-nmel',
            type=int,
            help='Number of melbands used for computing the melspectrogram.',
            default=128)
        parser.add_argument(
            '-scale',
            help=
            'Scale for the y-axis of the plots used by the system. Defaults to \'chroma\' in chroma mode.',
            default='linear',
            choices=['linear', 'log', 'mel'])
        parser.add_argument(
            '-delta',
            type=_check_positive,
            help=
            'If given, derivatives of the given order of the selected features are displayed in the plots used by the system.',
            default=None)
        return parser

    def extraction_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-net',
            help=
            'specify the CNN that will be used for the feature extraction. You need to specify a valid tensorflow model file in .pb format in your configuration file for this network.',
            default='AlexNet')
        parser.add_argument(
            '-layer',
            default='fc7',
            help=
            'name of CNN layer from which to extract the features.'
        )
        parser.add_argument(
            '-batch_size',
            type=int,
            help=
            'Maximum batch size for feature extraction. Adjust according to your gpu memory size.',
            default=128)
        return parser

    def writer_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-o',
            help=
            'the file which the features are written to. Supports csv and arff formats',
            required=True)
        parser.add_argument(
            '--no_timestamps',
            action='store_true',
            help='Remove timestamps from the output.')
        parser.add_argument(
            '--no_labels',
            action='store_true',
            help='Do not write class labels to the output.')
        return parser

    def label_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-l',
            help=
            'csv file with the labels for the files in the form: \'filename, label\'. If nothing is specified here or under -labels, the name(s) of the directory/directories are used as labels.',
            default=None)
        parser.add_argument(
            '-el',
            nargs=1,
            help='Define an explicit label for the input files.',
            default=None)
        return parser

    def parse_arguments(self):
        """
        Creates a commandline parser and handles the given options.
        :return: Nothing
        """
        self.parser = argparse.ArgumentParser(
            description='Extract deep spectrum features from wav files',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=self.parsers)

        args = vars(self.parser.parse_args())
        self.input = args['i']
        self.config = args['config']
        self.label_file = args['l']
        self.number_of_processes = args['np']

        # arguments for the plotting functions
        if self.plotting:
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
            self.plotting_args['output_spectrograms'] = abspath(
                args['specout']) if args['specout'] else None
            self.plotting_args['output_wavs'] = abspath(
                args['wavout']) if args['wavout'] else None

        # arguments for extraction functions
        if self.extraction:
            self.net = args['net']
            self.extraction_args['layer'] = args['layer']
            self.extraction_args['batch_size'] = args['batch_size']

        # arguments for writer
        if self.writer:
            self.writer_args['output'] = abspath(args['o'])
            makedirs(dirname(self.writer_args['output']), exist_ok=True)
            self.writer_args['continuous_labels'] = (
                                                            'window' in self.plotting_args
                                                    ) and args['tc'] and self.label_file
            self.writer_args['labels'] = args['el']
            self.writer_args['write_timestamps'] = (
                                                           args['t'][0] or args['t'][1]) and not args['no_timestamps']
            self.writer_args['no_labels'] = args['no_labels']

        # list all .wavs for the extraction found in the given folder
        self.files = sorted(self._find_files(self.input))
        if not self.files:
            self.parser.error(
                'No files were found. Check the specified input path.')

        print('Parsing labels...')
        if self.label_file is None:
            self._create_labels_from_folder_structure()
        else:
            self._read_label_file()

        self._load_config()

    def _find_files(self, path):
        if isfile(path):
            return [path]
        globexpression = '*.' + self.file_type
        reg_expr = re.compile(fnmatch.translate(globexpression), re.IGNORECASE)
        wavs = []
        for root, dirs, files in walk(path, topdown=True):
            wavs += [join(root, j) for j in files if re.match(reg_expr, j)]
        return wavs

    def _read_label_file(self):
        """
        Read labels from either .csv or .tsv files
        """

        # delimiters are decided by the extension of the labels file
        if self.label_file.endswith('.tsv'):
            parser = LabelParser(
                self.label_file,
                delimiter='\t',
                timecontinuous=self.writer_args['continuous_labels'])
        else:
            parser = LabelParser(
                self.label_file,
                delimiter=',',
                timecontinuous=self.writer_args['continuous_labels'])

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
        """
        if self.writer_args['labels'] is None:
            self.writer_args['label_dict'] = {
                basename(wav): [basename(dirname(wav))]
                for wav in self.files
            }
        else:
            # map the labels given on the commandline to all files in a given folder in the order both appear in the
            # parsed options.
            self.writer_args['label_dict'] = {
                basename(wav): self.writer_args['labels']
                for wav in self.files
            }
        labels = sorted(
            list(map(lambda x: x[0], self.writer_args['label_dict'].values())))
        self.writer_args['labels'] = [('class', set(labels))]

    def _load_config(self):
        """
        Parses the configuration file given on the commandline. If it does not exist yet, creates a new one containing
        standard settings.
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

            if self.extraction:
                print('Using tensorflow backend as specified in {}'.format(
                    self.config))
                self.extractor = TensorFlowExtractor
                net_conf = conf_parser['tensorflow-nets']

                if self.net in net_conf:
                    self.extraction_args['model_path'] = abspath(net_conf[
                                                                     self.net])
                else:
                    self.parser.error(
                        'No model weights defined for {} in {}'.format(
                            self.net, self.config))


def _check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue
