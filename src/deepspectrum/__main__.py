import rich_click as click
import logging
import logging.config

from deepspectrum.cli.features import features
from deepspectrum.cli.image_features import image_features
from deepspectrum.cli.plot import plot
from deepspectrum.cli.utils import add_options
from deepspectrum import __version__ as VERSION

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


_global_options = [
    click.option("-v", "--verbose", count=True),
]


version_str = (
    "DeepSpectrum %(version)s\nCopyright (C) 2017-2024 Shahin Amiriparian, Maurice Gerczuk, Sandra Ottl, "
    "Bjoern Schuller\n"
    "License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>.\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law."
)


@click.group()
@add_options(_global_options)
@click.version_option(VERSION, message=version_str)
def cli(verbose):
    click.echo("Verbosity: %s" % verbose)
    log_levels = ["ERROR", "INFO", "DEBUG"]
    verbose = min(2, verbose)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # this fixes the problem
            "formatters": {
                "standard": {"format": "%(asctime)s %(name)s: %(message)s"},
            },
            "handlers": {
                "default": {
                    "level": log_levels[verbose],
                    "class": "rich.logging.RichHandler",
                    "rich_tracebacks": True,
                    "formatter": "standard",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": log_levels[verbose],
                    "propagate": True,
                }
            },
        }
    )


cli.add_command(features)
cli.add_command(plot)
cli.add_command(image_features)
