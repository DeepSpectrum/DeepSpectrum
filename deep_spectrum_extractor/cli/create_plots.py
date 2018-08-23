
from os import environ


from tqdm import tqdm

import deep_spectrum_extractor.tools.feature_reduction as fr
from deep_spectrum_extractor.cli.configuration import Configuration
from ..backend.plotting import PlotGenerator


environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args=None):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration(extraction=False, writer=False, parser_description='Create plots from wav files.')
    configuration.parse_arguments()
    plots = PlotGenerator(
        files=configuration.files,
        number_of_processes=configuration.number_of_processes,
        **configuration.plotting_args)

    current_name = None
    with tqdm(total=len(plots), desc='Plotting wavs...') as pbar:

        for plot_tuple in plots:
            if current_name is None:
                current_name = plot_tuple.name
            elif current_name != plot_tuple.name:
                pbar.update()
                current_name = plot_tuple.name

    if configuration.reduced:
        print('Performing feature reduction...')
        fr.reduce_file(configuration.writer_args['output'],
                       configuration.reduced)


if __name__ == '__main__':
    main()

