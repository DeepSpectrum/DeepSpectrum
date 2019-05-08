import click
import logging
from os import environ
from tqdm import tqdm
from deepspectrum.cli.configuration import Configuration, GENERAL_OPTIONS, PLOTTING_OPTIONS
from .utils import add_options
from ..backend.plotting import PlotGenerator

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log = logging.getLogger(__name__)

DESCRIPTION_PLOT = 'Create plots from wav files.'


@click.command(help=DESCRIPTION_PLOT)
@add_options(GENERAL_OPTIONS)
@add_options(PLOTTING_OPTIONS)
def plot(**kwargs):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration(extraction=False, writer=False, **kwargs)
    plots = PlotGenerator(
        files=configuration.files,
        number_of_processes=configuration.number_of_processes,
        **configuration.plotting_args)

    current_name = None
    with tqdm(total=len(plots),
              desc='Plotting wavs...',
              disable=(log.getEffectiveLevel() >= logging.ERROR)) as pbar:

        for plot_tuple in plots:
            if current_name is None:
                current_name = plot_tuple.name
            elif current_name != plot_tuple.name:
                pbar.update()
                current_name = plot_tuple.name

    log.info('Done plotting wavs.')
