import click
import logging
import logging.config
import pkg_resources

from deepspectrum.cli.features import features
from deepspectrum.cli.image_features import image_features
from deepspectrum.cli.plot import plot
from deepspectrum.cli.utils import add_options

VERSION = pkg_resources.require("DeepSpectrum")[0].version

_global_options = [
    click.option('-v', '--verbose', count=True),
]


@click.group()
@add_options(_global_options)
@click.version_option(VERSION)
def cli(verbose):
    click.echo('Verbosity: %s' % verbose)
    log_levels = ['ERROR', 'INFO', 'DEBUG']
    verbose = min(2, verbose)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': log_levels[verbose],
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': log_levels[verbose],
                'propagate': True
            }
        }
    })


cli.add_command(features)
cli.add_command(plot)
cli.add_command(image_features)
