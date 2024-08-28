import logging
from os import environ

import rich_click as click
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from deepspectrum.cli.configuration import (
    GENERAL_OPTIONS,
    PLOTTING_OPTIONS,
    Configuration,
)

from ..backend.plotting import PlotGenerator
from .utils import add_options

environ["GLOG_minloglevel"] = "2"
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

log = logging.getLogger(__name__)

DESCRIPTION_PLOT = "Create plots from wav files."


@click.command(help=DESCRIPTION_PLOT)
@add_options(GENERAL_OPTIONS)
@add_options(PLOTTING_OPTIONS)
def plot(**kwargs):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration(extraction=False, writer=False, **kwargs)
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        plots = PlotGenerator(
            files=configuration.files,
            number_of_processes=configuration.number_of_processes,
            progress_display=progress,
            **configuration.plotting_args,
        )
        for _ in plots:
            pass

        log.info("Done plotting audios.")
