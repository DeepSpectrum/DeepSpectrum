import argparse
import csv
import io
from os import listdir, environ
from os.path import join, isfile, basename, expanduser, isdir
import configparser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imread import imread_from_blob
from scipy.io import wavfile
from tqdm import tqdm

environ['GLOG_minloglevel'] = '2'
import caffe


def get_wav_info(wav_file):
    frame_rate, sound_info = wavfile.read(wav_file)
    sound_info = np.trim_zeros(sound_info)
    return sound_info, frame_rate


def graph_spectrogram(wav_file, cmap):
    sound_info, frame_rate = get_wav_info(wav_file)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3, 3)
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    Pxx, freqs, bins, im = plt.specgram(sound_info, Fs=frame_rate, cmap=cmap)
    plt.xlim([0, len(sound_info) / frame_rate])
    plt.ylim([0, frame_rate / 2])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=129)
    buf.seek(0)
    plt.close("all")
    return buf.read()


def extract_features(img_blob, layer="fc7"):
    try:
        img = imread_from_blob(img_blob, 'png')
        img = caffe.io.skimage.img_as_float(img).astype(np.float32)
        img = img[:, :, :-1]
    except IOError:
        print('Error')
        return False, None
    # check if image is blank
    channel_0 = cv2.split(img)[0]
    channel_1 = cv2.split(img)[1]
    channel_2 = cv2.split(img)[2]
    non_zero_pixels = cv2.countNonZero(channel_0) + cv2.countNonZero(channel_1) + cv2.countNonZero(channel_2)
    if non_zero_pixels < 1:
        return False, None
    else:
        img = transformer.preprocess('data', img)
        net.blobs["data"].data[...] = img
        net.forward()
        return True, net.blobs[layer].data[0]


def write_features_to_csv(features, instance, instance_class, output_file, index=None, write_header=False):
    df = pd.DataFrame(features).T
    df.insert(0, "instance", instance)
    if index is not None:
        df.insert(1, "index", index)
    df.insert(len(df.columns), "class", instance_class)
    df.to_csv(output_file, index=False, mode='a', sep=",", header=write_header)


def create_label_dictionary(label_file):
    reader = csv.reader(open(label_file), delimiter="\t")
    dictionary = {}
    for row in reader:
        key = row[0]
        dictionary[key] = row[1]
    return dictionary


def batch_extract_folder(folder, output_file, labels=None, layer='fc7', cmap='viridis'):
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    files = sorted(files)
    print('Extracting features for ' + folder + ':')
    write_header = True
    for f in tqdm(files):
        filename = basename(f)
        img_blob = graph_spectrogram(f, cmap=cmap)
        success, features = extract_features(img_blob, layer=layer)
        if success:
            write_features_to_csv(features, filename, labels[filename], output_file, write_header=write_header)
            write_header = False


class Configuration:
    conf = {}
    model_def = ''
    model_weights = ''
    gpu_mode = True
    device_id = 0
    folder = ''
    output = ''
    cmap = ''
    labels = {}
    parser = argparse.ArgumentParser(description='Extract deep spectrum features from wav files')

    def __init__(self):
        self.init_parser()
        self.parse_arguments(vars(self.parser.parse_args()))
        self.configure_caffe()

    def init_parser(self):
        self.parser.add_argument('-f', help='folder where your wavs reside')
        self.parser.add_argument('-o', help='the file which the features are written to. Supports csv and arff formats')
        self.parser.add_argument('-l', help='csv file with the labels for the wavs in the form: test_001.wav  label')
        self.parser.add_argument('-cmap', default='viridis',
                            help='define the matplotlib colour map to use for the spectrograms')
        self.parser.add_argument('-config', help='path to configuration file which specifies caffe model and weight files',
                            default="deep.conf")

    def parse_arguments(self, args):
        self.folder = args['f']
        self.cmap = args['cmap']
        self.output = args['o']
        self.labels = create_label_dictionary(args['l'])
        self.load_config(args['config'])

    def load_config(self, conf_file):
        config = configparser.ConfigParser()
        if isfile(conf_file):
            config.read(conf_file)
            self.conf = config['main']
        else:
            home = expanduser('~')
            directory = join(home, 'caffe-master/models/bvlc_alexnet')
            self.conf = {'caffe_model_directory': directory,
                    'gpu': '1',
                    'device_id': '0'}
            config['main'] = self.conf
            with open(conf_file, 'w') as configfile:
                config.write(configfile)

    def configure_caffe(self):
        directory = self.conf['caffe_model_directory']
        try:
            self.model_def = [join(directory, file) for file in listdir(directory)
                              if file.endswith('deploy.prototxt')][0]
            self.model_weights = [join(directory, file) for file in listdir(directory)
                              if file.endswith('.caffemodel')][0]
            if isfile(self.model_weights):
                print('CaffeNet found.')
            else:
                # downloading net needs to be implemented
                print('Downloading pre-trained CaffeNet model...')

            if self.conf['gpu']=='1':
                caffe.set_device(int(self.conf['device_id']))
                caffe.set_mode_gpu()
            else:
                caffe.set_mode_cpu()
        except FileNotFoundError:
            print('Could not find model-directory. Check your configuration file!')


if __name__ == '__main__':
    config = Configuration()
    net = caffe.Net(config.model_def, caffe.TEST, weights=config.model_weights)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    net.blobs['data'].reshape(1, 3, 227, 227)
    net.reshape()

    batch_extract_folder(config.folder, config.output, config.labels, cmap=config.cmap)
