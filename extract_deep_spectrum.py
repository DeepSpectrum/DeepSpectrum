import argparse
import configparser
import csv
import io
from os import listdir, environ
from os.path import join, isfile, basename, expanduser, dirname, isdir

import matplotlib.pyplot as plt
import numpy as np
from imread import imread_from_blob
from scipy.io import wavfile
from tqdm import tqdm

environ['GLOG_minloglevel'] = '2'


import caffe


def get_wav_info(wav_file):
    frame_rate, sound_info = wavfile.read(wav_file)
    sound_info = np.trim_zeros(sound_info)
    return sound_info, frame_rate


def graph_spectrogram(wav_file, config):
    sound_info, frame_rate = get_wav_info(wav_file)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3, 3)
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    Pxx, freqs, bins, im = plt.specgram(sound_info, NFFT=config.nfft, Fs=frame_rate, cmap=config.cmap)
    extent = im.get_extent()
    plt.xlim([extent[0], extent[1]])
    plt.ylim([extent[2], extent[3]])
    # plt.xlim([0, len(sound_info) / frame_rate])
    # plt.ylim([0, frame_rate / 2])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=129)
    buf.seek(0)
    plt.close('all')
    return buf.read()


def graph_spectrogram_chunks(wav_file, config):
    sound_info, frame_rate = get_wav_info(wav_file)
    chunksize = int(config.chunksize / 1000 * frame_rate)
    step = chunksize if config.step is None else int(config.step / 1000 * frame_rate)
    chunks = [sound_info[n * step:min(n * step + chunksize, len(sound_info))] for n in
              range(int(len(sound_info) / step))]
    for chunk in chunks:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(3, 3)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax.set_axis_off()
        fig.add_axes(ax)
        Pxx, freqs, bins, im = plt.specgram(chunk, NFFT=config.nfft, noverlap=int(config.nfft / 2), Fs=frame_rate,
                                            cmap=config.cmap)
        extent = im.get_extent()
        plt.xlim([extent[0], extent[1]])
        plt.ylim([extent[2], extent[3]])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=129)
        buf.seek(0)
        plt.close('all')
        yield buf.read()


def extract_features(img_blob, config):
    try:
        img = imread_from_blob(img_blob, 'png')
        img = caffe.io.skimage.img_as_float(img).astype(np.float32)
        img = img[:, :, :-1]
    except IOError:
        print('Error')
        return False, None
    img = config.transformer.preprocess('data', img)
    config.net.blobs["data"].data[...] = img
    config.net.forward()
    return True, config.net.blobs[config.layer].data[0]


def batch_extract_folder(folder, config, writer):
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.wav')]
    files = sorted(files)
    print('Extracting features for ' + folder + ':')
    for f in tqdm(files):
        filename = basename(f)
        if config.chunksize is None:
            img_blob = graph_spectrogram(f, config)
            success, features = extract_features(img_blob, config)
            if success:
                writer.write(features, filename)
        else:
            for index, img_blob in tqdm(enumerate(graph_spectrogram_chunks(f, config)), leave=False, desc=filename):
                success, features = extract_features(img_blob, config)
                if success:
                    writer.write(features, filename, index=index)


def extract(config, writer):
    folders = [join(config.folder, subfolder) for subfolder in listdir(config.folder) if isdir(join(config.folder, subfolder))]
    if not folders:
        batch_extract_folder(config.folder, config, writer)
    else:
        for folder in folders:
            batch_extract_folder(folder, config, writer)
    writer.write_footer()


class FeatureWriter:
    def __init__(self, output, label_file=None):
        self.arff_mode = output.endswith('.arff')
        self.write_header = True
        self.output = output
        if label_file is None:
            print('No labels specified. Defaulting to \'?\'')
            self.label_dict = None
            self.labels = '?'
        else:
            self.label_dict, self.labels = self.read_label_file(label_file)

    @staticmethod
    def read_label_file(label_file):
        if label_file.endswith('.tsv'):
            reader = csv.reader(open(label_file), delimiter="\t")
        else:
            reader = csv.reader(open(label_file))
        dictionary = {}
        labels = set([])
        for row in reader:
            key = row[0]
            dictionary[key] = row[1]
            labels.add(dictionary[key])
        return dictionary, labels

    def write(self, features, name, index=None):
        if self.arff_mode:
            self.write_arff(features, name, index)
        else:
            self.write_csv(features, name, index)

    def write_csv(self, features, name, index=None):
        label = self.label_dict[name] if self.label_dict is not None else '?'
        if self.write_header:
            self.write_csv_header(features, index)
        with open(self.output, 'a') as f:
            if index is None:
                line = ','.join(map(str, [name] + features.tolist() + [label]))
            else:
                line = ','.join(map(str, [name] + [index] + features.tolist() + [label]))
            f.write(line + '\n')

    def write_csv_header(self, features, index):
        if index is None:
            header = ','.join(['name'] + ['neuron_' + str(i) for i in range(len(features))]+['class'])
        else:
            header = ','.join(['name'] + ['index'] + ['neuron_' + str(i) for i in range(len(features))]+['class'])
        with open(self.output, 'w') as f:
            f.write(header+'\n')
        self.write_header = False

    def write_arff(self, features, name, index=None):
        label = self.label_dict[name] if self.label_dict is not None else '?'
        if self.write_header:
            self.write_arff_header(features, index)
        with open(self.output, 'a') as f:
            if index is None:
                line = ','.join(map(str, [name] + features.tolist() + [label]))
            else:
                line = ','.join(map(str, [name] + [index] + features.tolist() + [label]))
            f.write(line + '\n')

    def write_arff_header(self, features, index):
        with open(self.output, 'w') as f:
            f.write('@RELATION \'Deep Spectrum Features\'\n\n@ATTRIBUTE name string\n')
            if index is not None:
                f.write('@ATTRIBUTE index numeric\n')
            for i, feature in enumerate(features):
                f.write('@ATTRIBUTE neuron_' + str(i) + ' numeric\n')
            f.write('@ATTRIBUTE class {' + ','.join(self.labels) + '}\n\n')
            f.write('@DATA\n')
        self.write_header = False

    def write_footer(self):
        if self.arff_mode:
            self.write_arff_footer(self.output)

    def write_arff_footer(self):
        with open(self.output, 'a') as f:
            f.write('%\n%\n%\n')


class Configuration:
    def __init__(self):
        # set default values
        self.conf = {}
        self.model_def = ''
        self.model_weights = ''
        self.gpu_mode = True
        self.device_id = 0
        self.folder = ''
        self.output = ''
        self.cmap = 'viridis'
        self.label_file = None
        self.layer = 'fc7'
        self.chunksize = None
        self.step = None
        self.nfft = 256

        # initialize commandline parser and parse the arguments
        self.parser = argparse.ArgumentParser(description='Extract deep spectrum features from wav files',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.requiredNamed = self.parser.add_argument_group('Required named arguments')
        self.init_parser()
        self.parse_arguments(vars(self.parser.parse_args()))

        # initialize the caffe net
        self.net = None
        self.transformer = None
        self.configure_caffe()

    def init_parser(self):

        self.requiredNamed.add_argument('-f', help='folder where your wavs reside', required=True)
        self.requiredNamed.add_argument('-o',
                                        help='the file which the features are written to. Supports csv and arff formats',
                                        required=True)
        self.parser.add_argument('-l',
                                 help='csv file with the labels for the wavs in the form: test_001.wav  label. If nothing is specified here, the name of the directory is used as label.',
                                 default=None)
        self.parser.add_argument('-cmap', default='viridis',
                                 help='define the matplotlib colour map to use for the spectrograms')
        self.parser.add_argument('-config',
                                 help='path to configuration file which specifies caffe model and weight files',
                                 default="deep.conf")
        self.parser.add_argument('-layer', default='fc7',
                                 help='name of CNN layer (as defined in caffe prototxt) from which to extract the features. Supports layers with 1-D output.')
        self.parser.add_argument('-chunksize', default=None, type=int,
                                 help='define a chunksize in ms. wav data is split into chunks of this length before feature extraction.')
        self.parser.add_argument('-step', default=None, type=int,
                                 help='stepsize for creating the wav segments in ms')
        self.parser.add_argument('-nfft', default=256,
                                 help='specify the size for the FFT window in number of samples', type=int)

    def parse_arguments(self, args):
        self.folder = args['f']
        self.cmap = args['cmap']
        self.output = args['o']
        self.label_file = args['l']
        if self.label_file is None:
            self.create_labels_from_folder_structure()
        self.load_config(args['config'])
        self.layer = args['layer']
        self.chunksize = args['chunksize']
        self.step = args['step']
        self.nfft = args['nfft']

    def create_labels_from_folder_structure(self):
        wavs = [join(self.folder, subfolder, wav_file) for subfolder in listdir(self.folder) for wav_file in
                listdir(join(self.folder, subfolder)) if
                isdir(join(self.folder, subfolder)) and isfile(join(self.folder, subfolder, wav_file))]
        dictionary = {basename(wav): basename(dirname(wav)) for wav in wavs}
        with open('labels.csv', 'w') as label_file:
            for key, value in dictionary.items():
                row = ','.join([key, value]) + '\n'
                label_file.write(row)
        self.label_file = 'labels.csv'

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
                         'device_id': '0'}
            conf_parser['main'] = self.conf
            with open(conf_file, 'w') as configfile:
                conf_parser.write(configfile)

    def configure_caffe(self):
        directory = self.conf['caffe_model_directory']
        try:
            self.model_def = [join(directory, file) for file in listdir(directory)
                              if file.endswith('deploy.prototxt')][0]
            self.model_weights = [join(directory, file) for file in listdir(directory)
                                  if file.endswith('.caffemodel')][0]
            if isfile(self.model_weights):
                print('CaffeNet found at ' + self.model_weights)
            else:
                print('Could not find CaffeNet.')

            if self.conf['gpu'] == '1':
                print('Using GPU-Mode')
                caffe.set_device(int(self.conf['device_id']))
                caffe.set_mode_gpu()
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
            print('Could not find model-directory. Check your configuration file!')


if __name__ == '__main__':
    configuration = Configuration()
    feature_writer = FeatureWriter(configuration.output, configuration.label_file)
    extract(configuration, feature_writer)
