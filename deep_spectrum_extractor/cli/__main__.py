import csv
import pathlib
from os import makedirs, environ

import numpy as np
from os.path import basename, join, dirname
from tqdm import tqdm

import deep_spectrum_extractor.backend.plotting as eds
import deep_spectrum_extractor.tools.feature_reduction as fr
from deep_spectrum_extractor.cli.configuration import Configuration
from deep_spectrum_extractor.tools.custom_arff import ArffWriter
from ..backend.plotting import PlotGenerator
from ..tools.feature_writer import get_writer

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args=None):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration()
    configuration.parse_arguments()
    plots = PlotGenerator(
        files=configuration.files,
        number_of_processes=configuration.number_of_processes,
        **configuration.plotting_args)

    print('Loading model and weights...')
    extractor = configuration.extractor(
            images=plots, **configuration.extraction_args)
    if configuration.extraction_args['layer'] not in extractor.layers:
        configuration.parser.error(
            '\'{}\' is not a valid layer name for {}. Available layers are: {}'.
            format(configuration.extraction_args['layer'], configuration.net,
                   extractor.layers))

    writer = get_writer(**configuration.writer_args)
    writer.write_features(configuration.files, extractor, hide_progress=False)

    if configuration.reduced:
        print('Performing feature reduction...')
        fr.reduce_file(configuration.writer_args['output'],
                       configuration.reduced)


if __name__ == '__main__':
    main()
