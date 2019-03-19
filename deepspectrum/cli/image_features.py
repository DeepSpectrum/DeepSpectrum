import cv2
import numpy as np
import click
from os import environ
from os.path import basename

import deepspectrum.tools.feature_reduction as fr
from .configuration import Configuration, GENERAL_OPTIONS, EXTRACTION_OPTIONS, LABEL_OPTIONS, WRITER_OPTIONS
from .ds_help import DESCRIPTION_IMAGE_FEATURES
from ..backend.plotting import PlotTuple
from ..tools.feature_writer import get_writer
from .utils import add_options

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

log = logging.getLogger(__name__)


def image_reader(files):
    for image in files:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = img[:, :, ::-1]
        yield PlotTuple(name=basename(image), timestamp=None, plot=np.array(img))

@click.command(help=DESCRIPTION_IMAGE_FEATURES)
@add_options(GENERAL_OPTIONS)
@add_options(EXTRACTION_OPTIONS)
@add_options(LABEL_OPTIONS)
@add_options(WRITER_OPTIONS)
def image_features(**kwargs):
    configuration = Configuration(plotting=False, file_type='png', **kwargs)
    plots = image_reader(configuration.files)
    log.info('Loading model and weights...')
    extractor = configuration.extractor(
        images=plots, **configuration.extraction_args)

    log.info('Extracting features from images...')
    writer = get_writer(**configuration.writer_args)
    writer.write_features(configuration.files, extractor)

    if configuration.reduced:
        log.info('Performing feature reduction...')
        fr.reduce_file(configuration.writer_args['output'],
                       configuration.reduced)


if __name__ == '__main__':
    main()
