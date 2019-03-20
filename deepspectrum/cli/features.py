import logging

log = logging.getLogger(__name__)

from os import environ

import click

import deepspectrum.tools.feature_reduction as fr

from deepspectrum.cli.configuration import Configuration, GENERAL_OPTIONS, PLOTTING_OPTIONS, EXTRACTION_OPTIONS, \
    LABEL_OPTIONS, WRITER_OPTIONS
from ..backend.plotting import PlotGenerator
from ..tools.feature_writer import get_writer
from .utils import add_options
from .ds_help import DESCRIPTION_EXTRACT

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@click.command(help=DESCRIPTION_EXTRACT)
@add_options(GENERAL_OPTIONS)
@add_options(PLOTTING_OPTIONS)
@add_options(EXTRACTION_OPTIONS)
@add_options(LABEL_OPTIONS)
@add_options(WRITER_OPTIONS)
def features(**kwargs):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration(plotting=True, extraction=True, writer=True,
                                  file_type='wav', **kwargs)
    plots = PlotGenerator(
        files=configuration.files,
        number_of_processes=configuration.number_of_processes,
        **configuration.plotting_args)

    log.info('Loading model and weights...')
    extractor = configuration.extractor(
        images=plots, **configuration.extraction_args)

    writer = get_writer(**configuration.writer_args)
    writer.write_features(configuration.files, extractor, hide_progress=False)

    if configuration.reduced:
        log.info('Performing feature reduction...')
        fr.reduce_file(configuration.writer_args['output'],
                       configuration.reduced)


if __name__ == '__main__':
    features()
