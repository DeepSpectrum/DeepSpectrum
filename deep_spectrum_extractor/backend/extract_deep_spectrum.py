import io
import warnings

import matplotlib

# force matplotlib to not use X-Windows backend. Needed for running the tool through an ssh connection.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from imread import imread_from_blob
from os import environ
from os.path import basename, join

environ['GLOG_minloglevel'] = '2'

import caffe


def _read_wav_data(wav_file, start=0, end=None):
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
    # sound_info = np.trim_zeros(sound_info)
    start = int(start * frame_rate)
    end = int(end * frame_rate) if end else None
    sound_info = sound_info[start:end] if end else sound_info[start:]
    return sound_info, frame_rate


def plot_spectrograms(wav_file, window, hop, nfft=256, cmap='viridis', size=227, output_folder=None,
                      y_limit=None, start=0, end=None):
    """
    Plot spectrograms for equally sized chunks of a wav-file using the described parameters.
    :param wav_file: path to an existing .wav file
    :param window: length of the chunks in s.
    :param hop: stepsize for chunking the audio data in s
    :param nfft: number of samples for the fast fourier transformation (Defaukt: 256)
    :param cmap: colourmap for the power spectral density (Default: 'viridis')
    :param size: size of the spectrogram plot in pixels. Height and width are alsways identical (Default: 227)
    :param output_folder: if given, the plot is saved to this existing folder in .png format (Default: None)
    :return: blob of the spectrogram plot
    """
    sound_info, frame_rate = _read_wav_data(wav_file, start=start, end=end)

    write_index = window or hop

    for idx, chunk in enumerate(_generate_chunks(sound_info, frame_rate, window, hop)):
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
            outfile = join(output_folder, file_name + '_' + str(idx) + '.png') if write_index else join(output_folder,
                                                                                                        file_name + '.png')
            plt.savefig(outfile, format='png', dpi=size)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=size)
        buf.seek(0)
        plt.close('all')
        yield buf.read()


def _generate_chunks(sound_info, sr, window, hop):
    if not window and not hop:
        return sound_info
    window = int(window * sr)
    hop = int(hop * sr)
    for n in range(max(int((len(sound_info)) / hop), 1)):
        yield sound_info[n * hop:min(n * hop + window, len(sound_info))]


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


def extract_features_from_wav(wav_file, input_transformer, caffe_net, nfft=256, layer='fc7', cmap='viridis', size=227,
                              chunksize=None, step=None, output_spectrograms=None, y_limit=None, start=0, end=None):
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
    for index, img_blob in enumerate(
            plot_spectrograms(wav_file, chunksize, step, nfft=nfft, cmap=cmap, size=size,
                              output_folder=output_spectrograms, y_limit=y_limit, start=start, end=end)):
        if chunksize:
            yield start + (index * step), extract_features_from_image_blob(img_blob, input_transformer, caffe_net,
                                                                           layer=layer)
        else:
            yield None, extract_features_from_image_blob(img_blob, input_transformer, caffe_net, layer=layer)
