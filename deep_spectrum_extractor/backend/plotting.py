import io
import warnings

import matplotlib


# force matplotlib to not use X-Windows backend. Needed for running the tool through an ssh connection.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile as sf
import deep_spectrum_extractor.models as models
from imread import imread_from_blob
from os import environ
from os.path import basename, join

environ['GLOG_minloglevel'] = '2'



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


def plot(wav_file, window, hop, mode='spectrogram', size=227, output_folder=None, start=0, end=None, data_spec=None, **kwargs):
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
            warnings.simplefilter('ignore')
            spectrogram_axes = PLOTTING_FUNCTIONS[mode](chunk, frame_rate, **kwargs)

        fig.add_axes(spectrogram_axes)

        if output_folder:
            file_name = basename(wav_file)[:-4]
            outfile = join(output_folder, file_name + '_' + str(idx) + '.png') if write_index else join(output_folder,
                                                                                                        file_name + '.png')
            fig.savefig(outfile, format='png', dpi=size)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=size)
        buf.seek(0)
        plt.close('all')
        img_blob = buf.read()
        try:
            img = imread_from_blob(img_blob, 'png')
            img = img[:, :, :-1]
        except IOError:
            print('Error while reading the spectrogram blob.')
            return None

        yield models.process_image(img, data_spec)



def plot_spectrogram(audio_data, sr, nfft=256, delta=None, **kwargs):
    spectrogram = librosa.stft(audio_data, n_fft=nfft, hop_length=int(nfft / 2))
    if delta:
        spectrogram = librosa.feature.delta(spectrogram, order=delta)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max, top_db=None)
    return _create_plot(spectrogram,sr, nfft, **kwargs)

def plot_mel_spectrogram(audio_data, sr, nfft=256, melbands=64, delta=None, **kwargs):
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=nfft,
                                                 hop_length=int(nfft / 2),
                                                 n_mels=melbands, power=1)
    if delta:
        spectrogram = librosa.feature.delta(spectrogram, order=delta)
    spectrogram = librosa.logamplitude(spectrogram, ref=np.max, top_db=None)
    return _create_plot(spectrogram, sr, nfft, **kwargs)

def plot_chroma(audio_data, sr, nfft=256, delta=None, **kwargs):
    spectrogram = librosa.feature.chroma_stft(audio_data, sr, n_fft=nfft, hop_length=int(nfft/2))
    if delta:
        spectrogram = librosa.feature.delta(spectrogram, order=delta)
    return _create_plot(spectrogram, sr, nfft, **kwargs)

def _create_plot(spectrogram, sr, nfft, y_limit=None, cmap='viridis', scale='linear'):
    if y_limit:
        relative_limit = y_limit * 2 / sr
        relative_limit = min(relative_limit, 1)
        spectrogram = spectrogram[:int(relative_limit * (1 + nfft / 2)), :]
    spectrogram_axes = librosa.display.specshow(spectrogram, hop_length=int(nfft / 2), sr=sr, cmap=cmap, y_axis=scale)
    return spectrogram_axes



def _generate_chunks(sound_info, sr, window, hop):
    if not window and not hop:
        yield sound_info
        return
    window = int(window * sr)
    hop = int(hop * sr)
    for n in range(max(int((len(sound_info)) / hop), 1)):
        yield sound_info[n * hop:min(n * hop + window, len(sound_info))]


PLOTTING_FUNCTIONS = {'spectrogram': plot_spectrogram,
                      'mel': plot_mel_spectrogram, 'chroma': plot_chroma}
