import matplotlib
matplotlib.use('Agg')
import io
import warnings
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import pathlib
import logging
from imread import imread_from_blob
from os import environ, makedirs
from os.path import basename, join, dirname, splitext
from multiprocessing import cpu_count, Pool
from functools import partial
from collections import namedtuple
from ..tools.path import get_relative_path

PlotTuple = namedtuple('PlotTuple', ['name', 'timestamp', 'plot'])
AudioChunk = namedtuple('AudioChunk',
                        ['name', 'samplerate', 'timestamp', 'audio'])

environ['GLOG_minloglevel'] = '2'

label_font = {'family': 'normal', 'size': 14}

font = {'family': 'normal', 'size': 12}

matplotlib.rc('font', **font)

log = logging.getLogger(__name__)


def read_wav_data(wav_file, start=0, end=None, resample=None):
    """
    Reads data from a wav-file and converts this data to single channel.
    :param wav_file: path to an existing .wav file
    :return: np array of audio data, frame rate
    """
    start = float(start) if start is not None else None
    end = float(end) if end is not None else None
    y, sr, = librosa.core.load(wav_file,
                               mono=True,
                               offset=start,
                               duration=end,
                               sr=resample)
    log.debug(f'Read audio file {wav_file}. Shape: {y.shape} Samplerate: {sr}')

    return y, sr


def plot_chunk(chunk,
               mode='spectrogram',
               output_folder=None,
               base_path=None,
               size=227,
               nfft=None,
               file_type='png',
               labelling=False,
               **kwargs):
    """
    Plot spectrograms for a chunk of a wav-file using the described parameters.
    :param chunk: audio chunk to be plotted.
    :param mode: type of audio plot to create.
    :param nfft: number of samples for the fast fourier transformation \
        (Default: 256)
    :param size: size of the spectrogram plot in pixels. Height and width are \
        always identical (Default: 227)
    :param output_folder: if given, the plot is saved to this path in .png \
        format (Default: None)
    :param kwargs: keyword args for plotting functions
    :return: blob of the spectrogram plot
    """
    filename, sr, ts, audio = chunk
    write_index = ts is not None
    if not nfft:
        nfft = _next_power_of_two(int(sr * 0.025))
    log.debug(f'Using nfft={nfft} for the FFT.')
    fig = plt.figure(frameon=False, tight_layout=False)

    if labelling:
        pass
    else:
        fig.set_size_inches(1, 1)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spectrogram_axes = PLOTTING_FUNCTIONS[mode](audio, sr, nfft, **kwargs)
        if labelling:
            original_xlim = spectrogram_axes.get_xlim()
            if mode != 'chroma':
                kHz_ticks = np.apply_along_axis(lambda x: x / 1000, 0,
                                                spectrogram_axes.get_yticks())
                spectrogram_axes.set_yticklabels(kHz_ticks)
                spectrogram_axes.set_ylabel('Frequency [kHz]',
                                            fontdict=label_font)
            else:
                spectrogram_axes.set_ylabel('Pitch Classes',
                                            fontdict=label_font)
            if labelling:
                spectrogram_axes.set_xticks(spectrogram_axes.get_xticks()[::2])
            spectrogram_axes.set_xlabel('Time [s]', fontdict=label_font)
            spectrogram_axes.set_xlim(original_xlim)
        del audio
    fig.add_axes(spectrogram_axes, id='spectrogram')

    if labelling:
        plt.colorbar(format='%+2.1f dB')
        plt.tight_layout()

    if output_folder:
        relative_file_name = f'{splitext(get_relative_path(filename, base_path))[0]}_{ts:g}.{file_type}' if write_index else f'{splitext(get_relative_path(filename, base_path))[0]}.{file_type}'
        if base_path is None:
            outfile = join(output_folder, basename(relative_file_name))
        else:
            outfile = join(output_folder, relative_file_name)

        log.debug(f'Saving spectrogram plot to {outfile}.')
        makedirs(dirname(outfile), exist_ok=True)
        fig.savefig(outfile, format=file_type, dpi=size)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=size)
    buf.seek(0)
    fig.clf()
    plt.close(fig)
    img_blob = buf.read()
    buf.close()
    try:
        img = imread_from_blob(img_blob, 'png')
        img = img[:, :, :-1]
        log.debug(f'Read spectrogram plot with shape {img.shape}.')
    except IOError:
        log.error('Error while reading the spectrogram blob.')
        return None
    return PlotTuple(name=get_relative_path(filename, base_path),
                     timestamp=ts,
                     plot=img)


def _generate_chunks_filename_timestamp_wrapper(filepath,
                                                window,
                                                hop,
                                                start=0,
                                                end=None,
                                                resample=None,
                                                nfft=256,
                                                wav_out_folder=None,
                                                base_path=None):
    sound_info, sr = read_wav_data(filepath,
                                   start=start,
                                   end=end,
                                   resample=resample)
    if not nfft:
        nfft = _next_power_of_two(int(sr * 0.025))
    if wav_out_folder is not None:
        relative_path = get_relative_path(filepath, base_path)
        wav_out = join(wav_out_folder, relative_path)
    else:
        wav_out = None
    for idx, audio in enumerate(
            _generate_chunks(sound_info, sr, window, hop, start, wav_out)):
        if window or hop:
            ts = start + idx * hop
        else:
            ts = None
        if len(audio) >= nfft:  # cannot plot chunks that are too short
            yield AudioChunk(filepath, sr, ts, audio)


def plot_spectrogram(audio_data, sr, nfft=None, delta=None, **kwargs):
    spectrogram = librosa.stft(audio_data,
                               n_fft=nfft,
                               hop_length=int(nfft / 2),
                               center=False)
    if delta:
        spectrogram = librosa.feature.delta(spectrogram, order=delta)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max, top_db=None)
    return _create_plot(spectrogram, sr, nfft, **kwargs)


def plot_mel_spectrogram(audio_data,
                         sr,
                         nfft=None,
                         melbands=64,
                         delta=None,
                         **kwargs):
    spectrogram = y_limited_spectrogram(audio_data,
                                        sr=sr,
                                        nfft=nfft,
                                        ylim=kwargs['ylim'])
    kwargs['scale'] = 'mel'
    if delta:
        spectrogram = librosa.feature.delta(spectrogram, order=delta)
    spectrogram = librosa.feature.melspectrogram(S=np.abs(spectrogram)**2,
                                                 sr=sr,
                                                 n_mels=melbands)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max, top_db=None)
    return _create_plot(spectrogram, sr, nfft, **kwargs)


def plot_chroma(audio_data, sr, nfft=None, delta=None, **kwargs):
    spectrogram = librosa.stft(audio_data,
                               n_fft=nfft,
                               hop_length=int(nfft / 2),
                               center=False)
    spectrogram = librosa.feature.chroma_stft(S=np.abs(spectrogram)**2, sr=sr)
    kwargs['scale'] = 'chroma'
    if delta:
        spectrogram = librosa.feature.delta(spectrogram, order=delta)
    return _create_plot(spectrogram, sr, nfft, **kwargs)


def y_limited_spectrogram(audio_data, sr, nfft=None, ylim=None):
    spectrogram = librosa.stft(audio_data,
                               n_fft=nfft,
                               hop_length=int(nfft / 2),
                               center=False)
    if ylim:
        relative_limit = ylim * 2 / sr
        relative_limit = min(relative_limit, 1)
        spectrogram = spectrogram[:int(relative_limit * (1 + nfft / 2)), :]
    return spectrogram


def _create_plot(spectrogram,
                 sr,
                 nfft,
                 ylim=None,
                 cmap='viridis',
                 scale='linear',
                 **kwargs):
    if not ylim:
        ylim = sr / 2
    spectrogram_axes = librosa.display.specshow(spectrogram,
                                                hop_length=int(nfft / 2),
                                                fmax=ylim,
                                                sr=sr,
                                                cmap=cmap,
                                                y_axis=scale,
                                                x_axis='time')
    if scale == 'linear':
        spectrogram_axes.set_ylim(0, ylim)

    return spectrogram_axes


PLOTTING_FUNCTIONS = {
    'spectrogram': plot_spectrogram,
    'mel': plot_mel_spectrogram,
    'chroma': plot_chroma
}


def _generate_chunks(sound_info, sr, window, hop, start=0, wav_out=None):
    if not window and not hop:
        yield sound_info
        return
    window_samples = int(window * sr)
    hop_samples = int(hop * sr)
    for n in range(max(int((len(sound_info)) / hop_samples), 1)):
        chunk = sound_info[n *
                           hop_samples:min(n * hop_samples +
                                           window_samples, len(sound_info))]
        if wav_out:
            makedirs(dirname(wav_out), exist_ok=True)
            chunk_out = f'{splitext(wav_out)[0]}_{(start + n * hop):g}.wav'
            librosa.output.write_wav(chunk_out, chunk, sr)
        yield chunk


def _next_power_of_two(x):
    return 1 << (x - 1).bit_length()


class PlotGenerator():
    def __init__(self,
                 files,
                 output_spectrograms=None,
                 output_wavs=None,
                 number_of_processes=None,
                 base_path=None,
                 **kwargs):
        self.files = files
        self.number_of_processes = number_of_processes
        if output_spectrograms:
            makedirs(output_spectrograms, exist_ok=True)
        if output_wavs:
            makedirs(output_wavs, exist_ok=True)
        if not self.number_of_processes:
            self.number_of_processes = cpu_count()
        self.chunks = (
            chunk for filename in self.files
            for chunk in _generate_chunks_filename_timestamp_wrapper(
                filename,
                wav_out_folder=output_wavs,
                window=kwargs['window'],
                hop=kwargs['hop'],
                start=kwargs['start'],
                end=kwargs['end'],
                nfft=kwargs['nfft'],
                resample=kwargs['resample'],
                base_path=base_path))
        plotting_func = partial(plot_chunk,
                                output_folder=output_spectrograms,
                                base_path=base_path,
                                **kwargs)

        self.pool = Pool(processes=self.number_of_processes)
        self.plots = self.pool.imap(plotting_func, self.chunks)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.plots)

        except StopIteration:
            self.pool.close()
            self.pool.join()
            raise StopIteration
