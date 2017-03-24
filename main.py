from os import listdir
from os.path import basename, isfile, join

from tqdm import tqdm

import extract_deep_spectrum_refactor as eds
import feature_reduction as fr
from configuration import Configuration
from feature_writer import FeatureWriter


def batch_extract_folder(folder, config, writer):
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.wav')]
    files = sorted(files)
    print('Extracting features for ' + folder + ':')
    for f in tqdm(files):
        filename = basename(f)
        if config.chunksize is None:
            img_blob = eds.graph_spectrogram(f, config)
            success, features = eds.extract_features(img_blob, config)
            if success:
                writer.write(features, filename)
        else:
            for index, img_blob in tqdm(enumerate(eds.graph_spectrogram_chunks(f, config)), leave=False, desc=filename):
                success, features = eds.extract_features(img_blob, config)
                if success:
                    writer.write(features, filename, index=index)


def batch_extract(config, writer):
    files = sorted(config.files)
    print('Extracting features...')
    for f in tqdm(files):
        filename = basename(f)
        if config.chunksize is None:
            img_blob = eds.graph_spectrogram(f, nfft=config.nfft, cmap=config.cmap, size=config.size)
            success, features = eds.extract_features(img_blob, config.transformer, config.net, layer=config.layer)
            if success:
                writer.write(features, filename)
        else:
            for index, img_blob in tqdm(enumerate(
                    eds.graph_spectrogram_chunks(f, config.chunksize, config.step, nfft=config.nfft, cmap=config.cmap,
                                                 size=config.size)), leave=False, desc=filename):
                success, features = eds.extract_features(img_blob, config.transformer, config.net, layer=config.layer)
                if success:
                    writer.write(features, filename, index=index)
    if config.reduced is not None:
        fr.reduce_features(writer.output, config.reduced)


if __name__ == '__main__':
    configuration = Configuration()
    feature_writer = FeatureWriter(configuration.output, configuration.label_dict)
    batch_extract(configuration, feature_writer)
