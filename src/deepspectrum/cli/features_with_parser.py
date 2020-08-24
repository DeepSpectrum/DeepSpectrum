import logging
import click
import importlib
from os import environ
from deepspectrum.cli.configuration import Configuration, GENERAL_OPTIONS,\
 PLOTTING_OPTIONS, EXTRACTION_OPTIONS, PARSER_OPTIONS, WRITER_OPTIONS, Filetypes
from deepspectrum.backend.extractor import _batch_images
from ..backend.plotting import PlotGenerator
from ..tools.feature_writer import get_writer
from .utils import add_options
from pathlib import Path
from os.path import splitext

from audeep.backend.parsers.meta import MetaParser
from audeep.backend.parsers.no_metadata import NoMetadataParser
from audeep.backend.data.data_set import Partition, Split



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
    num_folds = parser.num_folds
    partitions = set()
    if num_folds > 0:
        label_dicts = [{}]*num_folds
        for i in instances:
            nominal = i.label_nominal is not None
            fold = i.cv_folds.index(Split.VALID)
            label_dicts[fold][str(i.path)] = [i.label_nominal] if nominal else [i.label_numeric]    
    else:
        label_dicts = {'None': {}}
        for i in instances:
            nominal = i.label_nominal is not None
            if i.partition is None:
                label_dicts['None'][str(i.path)] = [i.label_nominal] if nominal else [i.label_numeric]
            else:
                if i.partition not in label_dicts:
                    partitions.add(i.partition)
                    label_dicts[i.partition] = {}
                label_dicts[i.partition][str(i.path)] = [i.label_nominal] if nominal else [i.label_numeric]

    use_folds = num_folds > 1
    use_partitions = len(partitions) > 1
    if nominal:
        labels = [("class", set(parser.label_map().keys()))]
    else:
        labels = [("label", "NUMERIC")]
            
            
    base_output = kwargs['output']
    extractor = None

    if use_partitions:
        for p in partitions:
            log_str = f"Extracting features for audio files in {kwargs['input']} using {parser.__class__.__name__}"
            output = base_output
           
            log_str += f" for partition {p.name.lower()}"
            output = splitext(output)[0] + f'.{p.name.lower()}' + splitext(output)[-1]
            kwargs['output'] = output
            log.info(log_str)
            label_dict = label_dicts[p]
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
            if extractor is None:
                extractor = configuration.extractor(images=plots,
                                                    **configuration.extraction_args)
            else: 
                extractor.set_images(plots)
            writer = get_writer(**configuration.writer_args)
            writer.write_features(configuration.files, extractor, hide_progress=False)
    elif use_folds:
        for i in range(num_folds):
            log_str = f"Extracting features for audio files in {kwargs['input']} using {parser.__class__.__name__}  for fold {i}"
            output = base_output
            output = splitext(output)[0] + f'.fold-{i}' + splitext(output)[-1]
            
            kwargs['output'] = output
            log.info(log_str)
            label_dict = label_dicts[i]
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
            if extractor is None:
                extractor = configuration.extractor(images=plots,
                                                    **configuration.extraction_args)
            else: 
                extractor.set_images(plots)

            writer = get_writer(**configuration.writer_args)
            writer.write_features(configuration.files, extractor, hide_progress=False)
    else: 
        log_str = f"Extracting features for audio files in {kwargs['input']} using {parser.__class__.__name__}"
        output = base_output
        kwargs['output'] = output
        log.info(log_str)
        label_dict = label_dicts['None']
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
