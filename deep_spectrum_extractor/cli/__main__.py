import csv
import multiprocessing as mp
import pathlib
from multiprocessing import JoinableQueue, Process
from os import makedirs, environ

import numpy as np
from os.path import basename, join, commonpath, dirname
from tqdm import tqdm

import deep_spectrum_extractor.backend.plotting as eds
import deep_spectrum_extractor.tools.feature_reduction as fr
from deep_spectrum_extractor.cli.configuration import Configuration
from deep_spectrum_extractor.tools.custom_arff import Writer

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def plotting_worker(config, job_queue, result_queue, **kwargs):
    # print('Process ' + str(id) + ': ' + 'initializing...')
    # wait for all threads to be initialized
    try:
        while True:
            file = job_queue.get()
            if file:
                plots = plot_file(file, config, **kwargs)
                if plots is not None:
                    filename = basename(file)
                    result_queue.put((filename, plots))
                job_queue.task_done()
            else:
                job_queue.task_done()
                # poison pilling for extractor
                result_queue.put((None, None))
                break
    except KeyboardInterrupt:
        pass

def plot_file(file, config, **kwargs):
    spectrogram_directory = None
    wav_directory = None
    if config.output_spectrograms:
        spectrogram_directory = join(config.output_spectrograms, get_relative_path(file, config.folders))
        makedirs(spectrogram_directory, exist_ok=True)
    if config.output_wavs:
        wav_directory = join(config.output_wavs, get_relative_path(file, config.folders))
        makedirs(wav_directory, exist_ok=True)
    return np.asarray([plot for plot in
                       eds.plot(file, output_folder=spectrogram_directory, wav_folder=wav_directory, **kwargs)])


def get_relative_path(file, folders):
    filepath = pathlib.PurePath(dirname(file))
    prefixes = [commonpath([file] + [folder]) for folder in folders]
    ml = max(len(s) for s in prefixes)
    prefix = list(set(s for s in prefixes if len(s) == ml))[0]
    folder_index = prefixes.index(prefix)
    filepath = filepath.relative_to(prefix)
    spectrogram_path = join('folder{}'.format(folder_index), str(filepath))
    return spectrogram_path


def writer_worker(total_num_of_files, number_of_processes, result_queue, start_condition, output, label_dict, labels, continuous_labels, window, hop, start, no_timestamps):

    with start_condition:
        start_condition.wait()
    print('Starting extraction with {} processes...'.format(number_of_processes))
    write_timestamp = window and not no_timestamps
    with tqdm(total=total_num_of_files) as pbar, open(output, 'w', newline='') as output_file:
        writer = None
        classes = [(class_name, '{' + ','.join(class_type) + '}') if class_type else (
            class_name, 'numeric') for class_name, class_type in labels]
        try:
            while True:
                file_name, features = result_queue.get()

                if features is not None:
                    file_name = basename(file_name)
                    for idx, feature_vector in enumerate(features):
                        if not writer:
                            attributes = _determine_attributes(write_timestamp, feature_vector, classes)
                            if output.endswith('.arff'):
                                writer = Writer(output_file, 'Deep Spectrum Features', attributes)
                            else:
                                writer = csv.writer(output_file, delimiter=',')
                                writer.writerow([attribute[0] for attribute in attributes])
                        if write_timestamp:
                            timestamp = start + (idx * hop)
                            labels = label_dict[file_name][write_timestamp] if continuous_labels else \
                                label_dict[file_name]
                            row = [file_name] + [str(timestamp)] + list(map(str, feature_vector)) + labels
                        else:
                            row = [file_name] + list(map(str, feature_vector)) + label_dict[file_name]
                        writer.writerow(row)
                    result_queue.task_done()
                    pbar.update(1)
                else:
                    result_queue.task_done()
                    break
        except KeyboardInterrupt:
            pass


def _determine_attributes(timestamp, feature_vector, classes):
    if timestamp:
        attributes = [('name', 'string'), ('timeStamp', 'numeric')] + [
            ('neuron_' + str(i), 'numeric') for i, _ in
            enumerate(feature_vector)] + classes
    else:
        attributes = [('name', 'string')] + [
            ('neuron_' + str(i), 'numeric') for i, _ in
            enumerate(feature_vector)] + classes
    return attributes



def main(args=None):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration()
    configuration.parse_arguments()

    file_name_queue = JoinableQueue()

    for file in configuration.files:
        file_name_queue.put(file)

    plot_queue = JoinableQueue()
    result_queue = JoinableQueue()
    total_num_of_files = len(configuration.files)
    number_of_processes = configuration.number_of_processes

    if not number_of_processes:
        number_of_processes = mp.cpu_count()

    start_writer = mp.Condition()
    processes = []
    for i in range(number_of_processes):
        p = Process(target=plotting_worker,
                    args=(configuration, file_name_queue, plot_queue), kwargs=configuration.plotting_args)
        p.daemon = True
        processes.append(p)
        file_name_queue.put(None)
        p.start()
    writer_thread = Process(target=writer_worker, args=(total_num_of_files, number_of_processes, result_queue, start_writer), kwargs=configuration.writer_args)
    writer_thread.daemon = True
    writer_thread.start()

    print('Loading model and weights...')
    if configuration.backend == 'caffe':
        from deep_spectrum_extractor.backend.extractor import CaffeExtractor
        extractor = CaffeExtractor(**configuration.extraction_args)
    elif configuration.backend == 'tensorflow':
        from deep_spectrum_extractor.backend.extractor import TensorFlowExtractor
        extractor = TensorFlowExtractor(**configuration.extraction_args)

    if configuration.extraction_args['layer'] not in extractor.layers:
        configuration.parser.error(
            '\'{}\' is not a valid layer name for {}. Available layers are: {}'.format(configuration.extraction_args['layer'], configuration.net, extractor.layers))

    with start_writer:
        start_writer.notify()

    finished_processes = 0
    try:
        while True:
            file_name, plots = plot_queue.get()
            if plots is not None:
                feature_vectors = extractor.extract_features(plots)
                result_queue.put((file_name, feature_vectors))
                plot_queue.task_done()
            else:
                finished_processes += 1
                if finished_processes == number_of_processes:
                    plot_queue.task_done()
                    result_queue.put((None, None))
                    break
                plot_queue.task_done()
    except KeyboardInterrupt:
        result_queue.put((None, None))

    for p in processes:
        p.join()
    writer_thread.join()
    plot_queue.close()
    result_queue.close()

    if configuration.reduced:
        print('Performing feature reduction...')
        fr.reduce_file(configuration.output, configuration.reduced)


if __name__ == '__main__':
    main()
