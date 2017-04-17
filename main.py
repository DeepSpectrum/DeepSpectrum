from os import listdir
from os.path import basename, isfile, join

from tqdm import tqdm

import extract_deep_spectrum as eds
import feature_reduction as fr
from configuration import Configuration
from feature_writer import FeatureWriter


def extract(config, writer):
    """
    Perform the extraction process with the parameter found in the configuration object.
    :param config: holds the extraction options
    :param writer: feature writer object
    :return: Nothing
    """
    files = sorted(config.files)
    print('Extracting features...')
    for f in tqdm(files):
        file_name = basename(f)
        for index, features in tqdm(eds.extract_features_from_wav(f, config.transformer, config.net,
                                                                             nfft=config.nfft,
                                                                             chunksize=config.chunksize,
                                                                             step=config.step, layer=config.layer,
                                                                             cmap=config.cmap, size=config.size,
                                                                             output_spectrograms=config.output_spectrograms),
                                               leave=False, desc=file_name):
            if features:
                writer.write(features, file_name, index=index)
    if config.reduced is not None:
        fr.reduce_features(writer.output, config.reduced)


if __name__ == '__main__':
    # set up the configuration object and parse commandline arguments
    configuration = Configuration()
    configuration.parse_arguments()

    # initialize feature writer and perform extraction
    feature_writer = FeatureWriter(configuration.output, configuration.label_dict)
    extract(configuration, feature_writer)
