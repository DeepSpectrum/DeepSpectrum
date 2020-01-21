import logging
import click
import importlib
from os import environ
from deepspectrum.cli.configuration import Configuration, GENERAL_OPTIONS,\
 PLOTTING_OPTIONS, EXTRACTION_OPTIONS, PARSER_OPTIONS, WRITER_OPTIONS, Filetypes
from ..backend.plotting import PlotGenerator
from ..tools.feature_writer import get_writer
from .utils import add_options
from pathlib import Path
from os.path import splitext

from audeep.backend.parsers.meta import MetaParser
from audeep.backend.parsers.no_metadata import NoMetadataParser
from audeep.backend.data.data_set import Partition



log = logging.getLogger(__name__)

DESCRIPTION_EXTRACT = 'Extract deep spectrum features from wav files.'

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@click.command(help=DESCRIPTION_EXTRACT)
@add_options(GENERAL_OPTIONS)
@add_options(PLOTTING_OPTIONS)
@add_options(EXTRACTION_OPTIONS)
@add_options(PARSER_OPTIONS)
@add_options(WRITER_OPTIONS)
def features_with_parser(**kwargs):
    # set up the configuration object and parse commandline arguments
    parser = kwargs.pop('parser')
    if parser is not None:
        module_name, class_name = parser.rsplit(".", 1)
        parser_class = getattr(importlib.import_module(module_name), class_name)
        if not parser_class(basedir=Path(kwargs['input'])).can_parse():
            log.error(f'Cannot parse dataset at {kwargs["input"]} using {parser}.')
            exit()
    
    else:
        parser_class = MetaParser
        if not parser_class(basedir=Path(kwargs['input'])).can_parse():
            parser_class = NoMetadataParser
        
    parser = parser_class(basedir=Path(kwargs['input']))
    instances = parser.parse()
    num_folds = parser.num_folds if parser.num_folds > 0 else 1
    partitions = set()
    label_dicts = {0: {}}
    for fold in range(num_folds):
        if not fold in label_dicts:
            label_dicts[fold] = {0: {}}
        for i in instances:
            if i.partition is not None:
                if not i.partition in label_dicts[fold]:
                    label_dicts[fold][i.partition] = {}
                    partitions.add(i.partition)
                if i.label_nominal is not None:
                    label_dicts[fold][i.partition][str(i.filename)] = [i.label_nominal]
                    nominal = True
                else:
                    label_dicts[fold][i.partition][str(i.filename)] = [i.label_numeric]
                    nominal = False

                
            else:
                if not Partition.TRAIN in label_dicts[fold]:
                    label_dicts[fold][Partition.TRAIN] = {}

                if i.label_nominal is not None:
                    label_dicts[fold][Partition.TRAIN][str(i.filename)] = [i.label_nominal]
                    nominal = True
                else:
                    label_dicts[fold][Partition.TRAIN][str(i.filename)] = [i.label_numeric]
                    nominal = False                
                    partitions.add(Partition.TRAIN)
    use_folds = num_folds > 1
    use_partitions = len(partitions) > 1
    if nominal:
        labels = [("class", set(parser.label_map.keys()))]
    else:
        labels = [("label", "NUMERIC")]
            
            
    base_output = kwargs['output']

    for f in range(num_folds):
        for p in partitions:
            log_str = f"Extracting features for audio files in {kwargs['input']} using {parser.__class__.__name__}"
            output = base_output
            if use_folds:
                log_str += f" for fold {f}"
                output = splitext(output)[0] + f'.fold-{f}' + splitext(output)[-1]
            if use_partitions:
                log_str += f" for partition {p.name.lower()}"
                output = splitext(output)[0] + f'.{p.name.lower()}' + splitext(output)[-1]
            kwargs['output'] = output
            log.info(log_str)
            label_dict = label_dicts[f][p]
            configuration = Configuration(plotting=True,
                                        extraction=True,
                                        writer=True,
                                        parser=True,
                                        label_dict=label_dict,
                                        labels=labels,
                                        file_type=Filetypes.AUDIO,
                                        **kwargs)
            plots = PlotGenerator(
                files=configuration.files,
                number_of_processes=configuration.number_of_processes,
                **configuration.plotting_args)

            log.info('Loading model and weights...')
            extractor = configuration.extractor(images=plots,
                                                **configuration.extraction_args)

            writer = get_writer(**configuration.writer_args)
            writer.write_features(configuration.files, extractor, hide_progress=False)

    log.info('Done extracting features.')
