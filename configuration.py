import argparse
import configparser
import csv
from os import listdir
from os.path import join, isfile, basename, expanduser, dirname, isdir

import caffe


class Configuration:
    def __init__(self):
        # set default values
        self.conf = {}
        self.model_def = ''
        self.model_weights = ''
        self.gpu_mode = True
        self.device_id = 0
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
        self.reduced = None
        self.size = 387
        self.files = []
        self.output_spectrograms = None

        # initialize commandline parser and parse the arguments
        self.parser = argparse.ArgumentParser(description='Extract deep spectrum features from wav files',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.requiredNamed = self.parser.add_argument_group('Required named arguments')
        self._init_parser()
        self._parse_arguments(vars(self.parser.parse_args()))

        # initialize the caffe net
        self.net = None
        self.transformer = None
        self.configure_caffe()

    def _init_parser(self):

        self.requiredNamed.add_argument('-f', nargs='+', help='folder(s) where your wavs reside', required=True)
        self.requiredNamed.add_argument('-o',
                                        help='the file which the features are written to. Supports csv and arff formats',
                                        required=True)
        self.parser.add_argument('-lf',
                                 help='csv file with the labels for the wavs in the form: \'test_001.wav, label\'. If nothing is specified here, the name(s) of the directory/directories are used as labels.',
                                 default=None)
        self.parser.add_argument('-cmap', default='viridis',
                                 help='define the matplotlib colour map to use for the spectrograms')
        self.parser.add_argument('-config',
                                 help='path to configuration file which specifies caffe model and weight files. If this file does not exist a new one is created and filled with the standard settings-',
                                 default="deep.conf")
        self.parser.add_argument('-layer', default='fc7',
                                 help='name of CNN layer (as defined in caffe prototxt) from which to extract the features. Supports layers with 1-D output.')
        self.parser.add_argument('-chunksize', default=None, type=int,
                                 help='define a chunksize in ms. wav data is split into chunks of this length before feature extraction.')
        self.parser.add_argument('-step', default=None, type=int,
                                 help='stepsize for creating the wav segments in ms')
        self.parser.add_argument('-nfft', default=256,
                                 help='specify the size for the FFT window in number of samples', type=int)
        self.parser.add_argument('-reduced', nargs='?',
                                 help='a reduced version of the feature set is written to the given location.',
                                 default=None, const='deep_spectrum_reduced.arff')
        self.parser.add_argument('-labels', nargs='+',
                                 help='define labels for folders explicitly in format: labelForFirstFolder labelForSecondFolder ...',
                                 default=None)
        self.parser.add_argument('-specout',
                                 help='define an existing folder where spectrogram plots should be saved during feature extraction. By default, spectrograms are not saved on disk to speed up extraction.',
                                 default=None)

    def _parse_arguments(self, args):
        self.folders = args['f']
        self.cmap = args['cmap']
        self.output = args['o']
        self.label_file = args['lf']
        self.labels = args['labels']
        self.files = [join(folder, wav_file) for folder in self.folders for wav_file in listdir(folder) if
                      isfile(join(folder, wav_file)) and (wav_file.endswith('.wav') or wav_file.endswith('.WAV'))]
        if not self.files:
            self.parser.error('No .wavs were found. Check the specified input paths.')
        if self.labels is not None and len(self.folders) != len(self.labels):
            self.parser.error(
                'Labels have to be specified for each folder: ' + str(len(self.folders)) + ' expected, ' + str(
                    len(self.labels)) + ' received.')
        print('Parsing labels...')
        if self.label_file is None:
            self.create_labels_from_folder_structure()
        else:
            self.read_label_file()
        self.load_config(args['config'])
        self.size = int(self.conf['size'])
        self.layer = args['layer']
        self.chunksize = args['chunksize']
        self.step = args['step'] if args['step'] else self.chunksize
        self.nfft = args['nfft']
        self.reduced = args['reduced']
        self.output_spectrograms = args['specout']
        if not isdir(self.output_spectrograms):
            self.parser.error('Spectrogram directory \''+self.output_spectrograms+'\' does not exist.')

    def read_label_file(self):
        if self.label_file.endswith('.tsv'):
            reader = csv.reader(open(self.label_file), delimiter="\t")
        else:
            reader = csv.reader(open(self.label_file))
        self.label_dict = {}
        self.labels = set([])
        for row in reader:
            key = row[0]
            self.label_dict[key] = row[1]
            self.labels.add(row[1])
        file_names = set(map(basename, self.files))
        missing_labels = file_names.difference(self.label_dict)
        if missing_labels:
            self.parser.error('No labels for: ' + ', '.join(missing_labels))

    def create_labels_from_folder_structure(self):
        if self.labels is None:
            wavs = [join(folder, wav_file) for folder in self.folders for wav_file in listdir(folder) if
                    isfile(join(folder, wav_file)) and (wav_file.endswith('.wav') or wav_file.endswith('.WAV'))]
            self.label_dict = {basename(wav): basename(dirname(wav)) for wav in wavs}
        else:
            self.label_dict = {wav: self.labels[folder_index] for folder_index, folder in enumerate(self.folders) for
                               wav in
                               listdir(folder) if
                               isfile(join(folder, wav)) and (wav.endswith('.wav') or wav.endswith('.WAV'))}

    def load_config(self, conf_file):
        conf_parser = configparser.ConfigParser()
        if isfile(conf_file):
            conf_parser.read(conf_file)
            self.conf = conf_parser['main']
        else:
            home = expanduser('~')
            directory = join(home, 'caffe-master/models/bvlc_alexnet')
            self.conf = {'caffe_model_directory': directory,
                         'gpu': '1',
                         'device_id': '0',
                         'size': '387'}
            conf_parser['main'] = self.conf
            with open(conf_file, 'w') as configfile:
                conf_parser.write(configfile)

    def configure_caffe(self):
        directory = self.conf['caffe_model_directory']
        try:
            model_defs = [join(directory, file) for file in listdir(directory) if file.endswith('deploy.prototxt')]
            if model_defs:
                self.model_def = model_defs[0]
                print('CaffeNet definition: ' + self.model_def)
            else:
                self.parser.error("No model definition found in " + directory + '.')
            model_weights = [join(directory, file) for file in listdir(directory)
                             if file.endswith('.caffemodel')]
            if model_weights:
                self.model_weights = model_weights[0]
                print('CaffeNet weights: ' + self.model_weights)
            else:
                self.parser.error("No model weights found in " + directory + '.')
            if self.conf['gpu'] == '1':
                caffe.set_device(int(self.conf['device_id']))
                caffe.set_mode_gpu()
                print('Using GPU device ' + int(self.conf['device_id']))
            else:
                print('Using CPU-Mode')
                caffe.set_mode_cpu()

            print('Loading Net')
            self.net = caffe.Net(self.model_def, caffe.TEST, weights=self.model_weights)
            self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            self.transformer.set_transpose('data', (2, 0, 1))
            self.transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
            self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
            shape = self.net.blobs['data'].shape
            self.net.blobs['data'].reshape(1, shape[1], shape[2], shape[3])
            self.net.reshape()
        except FileNotFoundError:
            raise
