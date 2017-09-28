import csv
import multiprocessing as mp
import pathlib
from multiprocessing import JoinableQueue, Process
from os import makedirs, environ

import numpy as np
import tensorflow as tf
from os.path import basename, join, commonpath, dirname
from tqdm import tqdm

import deep_spectrum_extractor.backend.plotting as eds
import deep_spectrum_extractor.models as models
import deep_spectrum_extractor.tools.feature_reduction as fr
from deep_spectrum_extractor.cli.configuration import Configuration
from deep_spectrum_extractor.tools.custom_arff import Writer

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# def extraction_worker(config, initialization, completed_inits, number_of_processes, job_queue, result_queue, gpu=0,
#                       id=0):
#     print('Process ' + str(id) + ': ' + 'initializing...')
#     directory = config.model_directory
#
#     # load model definition
#     model_def = config.model_def
#     model_weights = config.model_weights
#
#     # set mode to GPU or CPU computation
#     if config.gpu_mode:
#         caffe.set_device(gpu)
#         caffe.set_mode_gpu()
#         print('Process ' + str(id) + ': ' + 'Using GPU device ' + str(gpu))
#     else:
#         print('Process ' + str(id) + ': ' + 'Using CPU-Mode')
#         caffe.set_mode_cpu()
#
#     print('Process ' + str(id) + ': ' + 'Loading Net')
#     net = caffe.Net(model_def, caffe.TEST, weights=model_weights)
#     transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#     transformer.set_transpose('data', (2, 0, 1))
#     transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
#     transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
#
#     # reshape input layer as batch processing is not needed
#     shape = net.blobs['data'].shape
#     net.blobs['data'].reshape(1, shape[1], shape[2], shape[3])
#     net.reshape()
#
#     # wait for all threads to be initialized
#     with initialization:
#         completed_inits.value += 1
#         initialization.notify_all()
#         initialization.wait_for(lambda: completed_inits.value == number_of_processes.value, timeout=15)
#
#     while True:
#         file = job_queue.get()
#         if file:
#             features = [(i, fn, list(fv)) for i, fn, fv in extract_file(file, config, net, transformer)]
#             if features:
#                 result_queue.put(features)
#             job_queue.task_done()
#         else:
#             job_queue.task_done()
#             # poison pilling for writer
#             result_queue.put(None)
#             break

def plotting_worker(config, job_queue, result_queue,
                    data_spec=None, coordinator=None):
    # print('Process ' + str(id) + ': ' + 'initializing...')
    # wait for all threads to be initialized
    try:
        while not coordinator.should_stop():
            file = job_queue.get()
            if file:
                plots = plot_file(file, config, data_spec)
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
        coordinator.request_stop()


# def plot_file(file, config, net):
#     file_name = basename(file)
#     spectrogram_directory = None
#     if config.output_spectrograms:
#         spectrogram_directory = join(config.output_spectrograms, get_spectrogram_path(file, config.folders))
#         makedirs(spectrogram_directory, exist_ok=True)
#     for index, plot in eds.plot_spectrograms(file, config.window, config.hop, nfft=config.nfft, cmap=config.cmap, size=config.size, output_folder=spectrogram_directory,
#                       y_limit=config.y_limit, start=config.start, end=config.end, net=net):
#         yield index, file_name, plot

def plot_file(file, config, data_spec):
    spectrogram_directory = None
    if config.output_spectrograms:
        spectrogram_directory = join(config.output_spectrograms, get_spectrogram_path(file, config.folders))
        makedirs(spectrogram_directory, exist_ok=True)
    return np.asarray([plot for plot in
                       eds.plot(file, config.window, config.hop, scale=config.scale, mode=config.mode,
                                delta=config.delta, nfft=config.nfft, cmap=config.cmap,
                                size=config.size, output_folder=spectrogram_directory,
                                y_limit=config.y_limit, start=config.start, end=config.end,
                                data_spec=data_spec)])


def get_spectrogram_path(file, folders):
    filepath = pathlib.PurePath(dirname(file))
    prefixes = [commonpath([file] + [folder]) for folder in folders]
    ml = max(len(s) for s in prefixes)
    prefix = list(set(s for s in prefixes if len(s) == ml))[0]
    folder_index = prefixes.index(prefix)
    filepath = filepath.relative_to(prefix)
    spectrogram_path = join('folder{}'.format(folder_index), str(filepath))
    return spectrogram_path


def writer_worker(config, total_num_of_files, number_of_processes, result_queue, coordinator, start_condition):
    # with initialization:
    #     initialization.wait_for(lambda: completed_inits.value == number_of_processes.value, timeout=15)
    with start_condition:
        start_condition.wait()
    print('Starting extraction with {} processes...'.format(number_of_processes))
    write_timestamp = config.window or config.hop
    with tqdm(total=total_num_of_files) as pbar, open(config.output, 'w', newline='') as output_file:
        writer = None
        classes = [(class_name, '{' + ','.join(class_type) + '}') if class_type else (
            class_name, 'numeric') for class_name, class_type in config.labels]
        try:
            while not coordinator.should_stop():
                file_name, features = result_queue.get()

                if features is not None:
                    file_name = basename(file_name)
                    for idx, feature_vector in enumerate(features):
                        if not writer:
                            attributes = _determine_attributes(write_timestamp, feature_vector, classes)
                            if config.output.endswith('.arff'):
                                writer = Writer(output_file, 'Deep Spectrum Features', attributes)
                            else:
                                writer = csv.writer(output_file, delimiter=',')
                                writer.writerow([attribute[0] for attribute in attributes])
                        if write_timestamp:
                            timestamp = config.start + (idx * config.hop)
                            labels = config.label_dict[file_name][write_timestamp] if config.continuous_labels else \
                                config.label_dict[file_name]
                            row = [file_name] + [str(timestamp)] + list(map(str, feature_vector)) + labels
                        else:
                            row = [file_name] + list(map(str, feature_vector)) + config.label_dict[file_name]
                        writer.writerow(row)
                    result_queue.task_done()
                    pbar.update(1)
                else:
                    result_queue.task_done()
                    break
        except KeyboardInterrupt:
            pass
        finally:
            coordinator.request_stop()


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


def kill_inactive(process_list):
    for process in process_list:
        if not process.is_alive():
            process.terminate()


# def spectrogram_batch(filename, capacity, batch_size=10, configuration=None, net=None):
#     plotting_func = partial(plot_file, config=configuration, net=net)
#     data = filename, tf.py_func(plotting_func, filename, [tf.float32])
#     # Create a FIFO shuffle queue.
#     queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string, tf.float32])
#     # Create an op to enqueue one item.
#     enqueue = queue.enqueue(data)
#
#     # Create a queue runner that, when started, will launch 4 threads applying
#     # that enqueue op.
#     num_threads = 4
#     qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)
#
#     # Register the queue runner so it can be found and started by
#     # `tf.train.start_queue_runners` later (the threads are not launched yet).
#     tf.train.add_queue_runner(qr)
#
#     # Create an op to dequeue a batch
#     return queue.dequeue()


def main(args=None):
    # set up the configuration object and parse commandline arguments
    configuration = Configuration()
    configuration.parse_arguments()
    net = models.load_model(configuration.net)
    print('Using {} with weights found in {}.'.format(net.__class__.__name__, configuration.model_weights))
    if configuration.layer not in net.layers:
        configuration.parser.error(
            '\'{}\' is not a valid layer name for {}. Available layers are: {}'.format(configuration.layer,
                                                                                       net.__class__.__name__,
                                                                                       list(net.layers.keys())))
    data_spec = models.get_data_spec(model_instance=net)

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
    coordinator = tf.train.Coordinator()
    processes = []
    for i in range(number_of_processes):
        p = Process(target=plotting_worker,
                    args=(configuration, file_name_queue, plot_queue, data_spec, coordinator))
        p.daemon = True
        processes.append(p)
        file_name_queue.put(None)
        p.start()
    writer_thread = Process(target=writer_worker, args=(
        configuration, total_num_of_files, number_of_processes, result_queue, coordinator, start_writer))
    writer_thread.daemon = True
    writer_thread.start()

    # filename_queue = tf.train.string_input_producer(configuration.files, num_epochs=1,
    #                                                 capacity=len(configuration.files))
    # filename = filename_queue.dequeue()
    # get_batch = spectrogram_batch([filename], capacity=100000, configuration=configuration, net=net)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print('Loading model and weights...')
        net.load(configuration.model_weights, sess)
        with start_writer:
            start_writer.notify()
        # tf.local_variables_initializer().run()
        # threads = tf.train.start_queue_runners(sess, coord=coordinator)
        finished_processes = 0
        try:
            while not coordinator.should_stop():
                file_name, plots = plot_queue.get()
                if plots is not None:
                    feature_vectors = sess.run(net.layers[configuration.layer], feed_dict={net.layers['data']: plots})
                    result_queue.put((file_name, feature_vectors))
                    plot_queue.task_done()
                else:
                    finished_processes += 1
                    if finished_processes == number_of_processes:
                        plot_queue.task_done()
                        result_queue.put((None, None))
                        break
                    plot_queue.task_done()

        except tf.errors.OutOfRangeError:
            result_queue.put((None, None))
        except KeyboardInterrupt:
            result_queue.put((None, None))
        finally:
            coordinator.request_stop()

    coordinator.join(processes)
    writer_thread.join()
    plot_queue.close()
    result_queue.close()

    # for i in range(number_of_processes.value):
    #
    #     p = Process(target=plotting_worker,
    #                 args=(configuration, initialization, completed_inits, number_of_processes, job_queue, plot_queue, i, net))
    #     p.daemon = True
    #     processes.append(p)
    #     p.start()
    #
    # writer_thread = Process(target=writer_worker, args=(
    #     configuration, initialization, completed_inits, total_num_of_files, number_of_processes, plot_queue, net))
    # writer_thread.daemon = True
    # writer_thread.start()
    #
    # with initialization:
    #     initialization.wait_for(lambda: completed_inits.value == number_of_processes.value, timeout=15)
    #     number_of_processes.value = sum(map(lambda x: x.is_alive(), processes))
    #
    # # kill dead processes
    # kill_inactive(processes)
    #
    # # poison pilling for extraction workers
    # for i in range(number_of_processes.value):
    #     job_queue.put(None)
    #
    # job_queue.join()

    #
    if configuration.reduced:
        print('Performing feature reduction...')
        fr.reduce_file(configuration.output, configuration.reduced)


if __name__ == '__main__':
    main()
