from os import listdir
from os.path import basename, isfile, join

from tqdm import tqdm

import extract_deep_spectrum as eds
import feature_reduction as fr
from configuration import Configuration
from feature_writer import FeatureWriter


def batch_extract(config, writer):
    files = sorted(config.files)
    print('Extracting features...')
    for f in tqdm(files):
        filename = basename(f)
        if config.chunksize is None:
            img_blob = eds.graph_spectrogram(f, nfft=config.nfft, cmap=config.cmap, size=config.size)
            success, features = eds.extract_features_from_image_blob(img_blob, config.transformer, config.net,
                                                                     layer=config.layer)
            if success:
                writer.write(features, filename)
        else:
            for index, img_blob in tqdm(enumerate(
                    eds.graph_spectrogram_chunks(f, config.chunksize, config.step, nfft=config.nfft, cmap=config.cmap,
                                                 size=config.size)), leave=False, desc=filename):
                success, features = eds.extract_features_from_image_blob(img_blob, config.transformer, config.net,
                                                                         layer=config.layer)
                if success:
                    writer.write(features, filename, index=index)
    if config.reduced is not None:
        fr.reduce_features(writer.output, config.reduced)


def extract(config, writer):
    files = sorted(config.files)
    print('Extracting features...')
    for f in tqdm(files):
        file_name = basename(f)
        for index, (success, features) in eds.extract_features_from_wav(f, config.transformer, config.net,
                                                                      nfft=config.nfft, chunksize=config.chunksize,
                                                                      step=config.step, layer=config.layer,
                                                                      cmap=config.cmap, size=config.size, output_spectrograms=config.output_spectrograms):
            if success:
                writer.write(features, file_name, index=index)


if __name__ == '__main__':
    configuration = Configuration()
    feature_writer = FeatureWriter(configuration.output, configuration.label_dict)
    extract(configuration, feature_writer)
