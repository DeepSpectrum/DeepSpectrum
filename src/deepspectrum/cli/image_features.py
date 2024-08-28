import logging
from os import environ, curdir

import click
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..backend.plotting import PlotTuple
from os.path import relpath
from .configuration import (
    EXTRACTION_OPTIONS,
    GENERAL_OPTIONS,
    LABEL_OPTIONS,
    WRITER_OPTIONS,
    Configuration,
    Filetypes,
)
from .utils import add_options

environ["GLOG_minloglevel"] = "2"
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

log = logging.getLogger(__name__)

DESCRIPTION_IMAGE_FEATURES = "Extract CNN-descriptors from images."


def image_reader(files, base_path=curdir, size=500):
    import cv2

    for image in files:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(size, size))
        img = img[:, :, :3]
        yield PlotTuple(
            name=relpath(image, start=base_path), timestamp=None, plot=np.array(img)
        )


@click.command(help=DESCRIPTION_IMAGE_FEATURES)
@add_options(GENERAL_OPTIONS)
@add_options(EXTRACTION_OPTIONS)
@add_options(LABEL_OPTIONS)
@add_options(WRITER_OPTIONS[:-2])
def image_features(**kwargs):
    from ..tools.feature_writer import get_writer

    configuration = Configuration(plotting=False, file_type=Filetypes.IMAGE, **kwargs)
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        plots = image_reader(configuration.files, base_path=configuration.input_folder)
        log.info("Loading model and weights...")
        extractor = configuration.extractor(
            images=plots, **configuration.extraction_args
        )

        log.info("Extracting features from images...")
        writer = get_writer(**configuration.writer_args, progress_display=progress)
        writer.write_features(configuration.files, extractor)

        log.info("Done extracting features.")
