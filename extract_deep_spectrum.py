import io
import warnings
from math import ceil, log2
from os import environ
from os.path import basename, join
from PIL import Image

import cv2
import matplotlib

# force matplotlib to not use X-Windows backend. Needed for running the tool through an ssh connection.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from imread import imread_from_blob

environ['GLOG_minloglevel'] = '2'

import caffe


def _read_wav_data(wav_file):
    """
    Reads data from a wav-file, converts this data to single channel and trims zeros at beginning and end of the
    audio data.
    :param wav_file: path to an existing .wav file
    :return: np array of audio data, frame rate
    """
    sound_info, frame_rate, = sf.read(wav_file)
    # convert stereo to mono
    if len(sound_info.shape) > 1:
        sound_info = sound_info.astype(float)
        sound_info = sound_info.sum(axis=1) / 2
        sound_info = np.array(sound_info)
    # remove zeros at beginning and end of audio file
    sound_info = np.trim_zeros(sound_info)
    return sound_info, frame_rate


def plot_spectrogram(wav_file, nfft=256, cmap='viridis', size=227, output_folder=None, y_limit=None):
    """
    Plots a spectrogram from a given .wav file using the described parameters.
    :param wav_file: path to an existing .wav file
    :param nfft: number of samples for the fast fourier transformation (Defaukt: 256)
    :param cmap: colourmap for the power spectral density (Default: 'viridis')
    :param size: size of the spectrogram plot in pixels. Height and width are alsways identical (Default: 227)
    :param output_folder: if given, the plot is saved to this existing folder in .png format (Default: None)
    :return: blob of the spectrogram plot
    """
    sound_info, frame_rate = _read_wav_data(wav_file)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)

    # set figure size
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Pxx, freqs, bins, im = plt.specgram(sound_info, NFFT=nfft, Fs=frame_rate, cmap=cmap, noverlap=int(nfft / 2))

    # limit figure to plot
    extent = im.get_extent()
    plt.xlim([extent[0], extent[1]])

    if y_limit:
        plt.ylim([extent[2], y_limit])
    else:
        plt.ylim([extent[2], extent[3]])

    if output_folder:
        file_name = basename(wav_file)[:-4]
        plt.savefig(join(output_folder, file_name + '.png'), format='png', dpi=size)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=size)
    buf.seek(0)
    plt.close('all')
    return buf.read()


def plot_spectrogram_chunks(wav_file, chunksize, step, nfft=256, cmap='viridis', size=227, output_folder=None,
                            y_limit=None):
    """
    Plot spectrograms for equally sized chunks of a wav-file using the described parameters.
    :param wav_file: path to an existing .wav file
    :param chunksize: length of the chunks in ms
    :param step: stepsize for chunking the audio data. Defaults to the given chunksize
    :param nfft: number of samples for the fast fourier transformation (Default: 256)
    :param cmap: colourmap for the power spectral density (Default: 'viridis')
    :param size: size of the spectrogram plot in pixels. Height and width are alsways identical (Default: 227)
    :param output_folder: if given, the plot is saved to this existing folder in .png format (Default: None)
    :return: blob of the spectrogram plot
    """
    sound_info, frame_rate = _read_wav_data(wav_file)

    # size of chunks in number of samples
    chunksize = int(chunksize / 1000 * frame_rate)
    step = chunksize if step is None else int(step / 1000 * frame_rate)

    # list chunks from the audio data
    chunks = [sound_info[n * step:min(n * step + chunksize, len(sound_info))] for n in
              range(max(int((len(sound_info)) / step), 1))]
    for idx, chunk in enumerate(chunks):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1, 1)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax.set_axis_off()
        fig.add_axes(ax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Pxx, freqs, bins, im = plt.specgram(chunk, NFFT=nfft, noverlap=int(nfft / 2), Fs=frame_rate,
                                                cmap=cmap)
        extent = im.get_extent()

        plt.xlim([extent[0], extent[1]])

        if y_limit:
            plt.ylim([extent[2], y_limit])
        else:
            plt.ylim([extent[2], extent[3]])

        if output_folder:
            file_name = basename(wav_file)[:-4]
            plt.savefig(join(output_folder, file_name + '_' + str(idx) + '.png'), format='png', dpi=size)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=size)
        buf.seek(0)
        plt.close('all')
        yield buf.read()


def extract_features_from_image_blob(img_blob, input_transformer, caffe_net, layer='fc7'):
    """
    Extracts deep features from a given image blob using a pre-loaded Caffe Network. Features from high dimenional
    layers are flattened to 1D.
    :param img_blob: blob containing image data in .png format
    :param input_transformer: configured transformer object belonging to a loaded Caffe Network
    :param caffe_net: a pre loaded Caffe Network used for feature extraction
    :param layer: name of the extraction layer as given in the network's deploy.prototxt
    :return: extracted feature vector
    """
    try:
        img = imread_from_blob(img_blob, 'png')
        img = caffe.io.skimage.img_as_float(img).astype(np.float32)
        if img.shape[2] == 4:
            img = img[:, :, :-1]
    except IOError:
        print('Error while reading the spectrogram blob.')
        return None

    # pre process image data and forward it through the loaded CNN network
    img = input_transformer.preprocess('data', img)
    caffe_net.blobs["data"].data[...] = img
    caffe_net.forward()

    # extract features from the specified layer
    features = caffe_net.blobs[layer].data[0]
    return np.ravel(features)


def scale_image(image, scale, net):
    iter_all = ceil(log2(scale))

    img = imread_from_blob(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)

    for i in range(iter_all):
        net.blobs['data'].reshape(1, 1, img.shape[0]*2, img.shape[1]*2)
        net.reshape()

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_raw_scale('data', 1 / 255.0)  # rescale from [0, 1] to [0, 255]

        img = transformer.preprocess('data', img)
        net.blobs['data'].data[...] = img[0, :, :]
        net.forward()
        img0 = net.blobs["sum"].data[0, :, :, :]
        img[0] = img0

        img = transformer.deprocess('data', img)
        np.clip(img, 0, 255, out=img)
        img = img.astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
    img = Image.fromarray(img, 'RGB')
    buf = io.BytesIO()
    img.save(buf, 'png')
    buf.seek(0)
    return buf.read()


def extract_features_from_wav(wav_file, input_transformer, caffe_net, nfft=256, layer='fc7', cmap='viridis', size=227,
                              chunksize=None, step=None, output_spectrograms=None, y_limit=None, scale=None, scaling_net=None):
    """
    Extracts deep spectrum features from a given wav-file using either the whole file or equally sized chunks as basis
    for the spectrogram plots.
    :param wav_file: path to an existing .wav file
    :param input_transformer: configured transformer object belonging to a loaded Caffe Network
    :param caffe_net: a pre loaded Caffe Network used for feature extraction
    :param nfft: number of samples for the fast fourier transformation (Defaukt: 256)
    :param layer: name of the extraction layer as given in the network's deploy.prototxt
    :param cmap: colourmap for the power spectral density (Default: 'viridis')
    :param size: size of the spectrogram plot in pixels. Height and width are alsways identical (Default: 227)
    :param chunksize: length of the chunks in ms
    :param step: stepsize for chunking the audio data. Defaults to the given chunksize
    :param output_spectrograms: if given, the plot is saved to this existing folder in .png format (Default: None)
    :return: index of the extracted feature vector (in the case of extracting from chunks), feature vector
    """

    # just extract a single vector
    if not chunksize:
        img_blob = plot_spectrogram(wav_file, nfft=nfft, cmap=cmap, size=size, output_folder=output_spectrograms,
                                    y_limit=y_limit)
        if scale and scaling_net:
            img_blob = scale_image(img_blob, scale, scaling_net)
        yield None, extract_features_from_image_blob(img_blob, input_transformer, caffe_net, layer=layer)
        return

    # extract features for chunks of the wav-file
    else:
        for index, img_blob in enumerate(
                plot_spectrogram_chunks(wav_file, chunksize, step, nfft=nfft, cmap=cmap, size=size,
                                        output_folder=output_spectrograms, y_limit=y_limit)):
            if scale and scaling_net:
                img_blob = scale_image(img_blob, scale, scaling_net)
            yield index, extract_features_from_image_blob(img_blob, input_transformer, caffe_net, layer=layer)
