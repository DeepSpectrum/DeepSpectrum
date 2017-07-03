import pathlib
import multiprocessing as mp
from multiprocessing import JoinableQueue, Process, Value, Condition
from os import makedirs, listdir, environ
from os.path import basename, join, commonpath, dirname

from tqdm import tqdm

import extract_deep_spectrum as eds
import feature_reduction as fr
from configuration import Configuration
from feature_writer import FeatureWriter

environ['GLOG_minloglevel'] = '2'

import caffe


def extraction_worker(config, gpu=0, id=0):
    print('Process ' + str(id) + ': ' + 'initializing...')
    directory = config.model_directory

    # load model definition
    model_def = config.model_def
    model_weights = config.model_weights

    # set mode to GPU or CPU computation
    if config.gpu_mode:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        print('Process ' + str(id) + ': ' + 'Using GPU device ' + str(gpu))
    else:
        print('Process ' + str(id) + ': ' + 'Using CPU-Mode')
        caffe.set_mode_cpu()

    print('Process ' + str(id) + ': ' + 'Loading Net')
    net = caffe.Net(model_def, caffe.TEST, weights=model_weights)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # reshape input layer as batch processing is not needed
    shape = net.blobs['data'].shape
    net.blobs['data'].reshape(1, shape[1], shape[2], shape[3])
    net.reshape()

    # wait for all threads to be initialized
    with initialization:
        completed_inits.value += 1
        initialization.notify_all()
        initialization.wait_for(lambda: completed_inits.value == number_of_processes.value, timeout=15)

    while True:
        file = job_queue.get()
        if file:
            features = [(i, fn, list(fv)) for i, fn, fv in extract_file(file, config, net, transformer)]
            if features:
                result_queue.put(features)
            job_queue.task_done()
        else:
            job_queue.task_done()
            #poison pilling for writer
            result_queue.put(None)
            break


def extract_file(file, config, net, transformer):
    file_name = basename(file)
    spectrogram_directory = None
    if config.output_spectrograms:
        spectrogram_directory = join(config.output_spectrograms, get_spectrogram_path(file, config.folders))
        makedirs(spectrogram_directory, exist_ok=True)
    for index, features in eds.extract_features_from_wav(file, transformer, net,
                                                         nfft=config.nfft,
                                                         chunksize=config.chunksize,
                                                         step=config.step, layer=config.layer,
                                                         cmap=config.cmap, size=config.size,
                                                         output_spectrograms=spectrogram_directory, y_limit=config.y_limit):
        if features.any():
            yield index, file_name, features


def get_spectrogram_path(file, folders):
    filepath = pathlib.PurePath(dirname(file))
    prefixes = [commonpath([file] + [folder]) for folder in folders]
    ml = max(len(s) for s in prefixes)
    prefix = list(set(s for s in prefixes if len(s) == ml))[0]
    folder_index = prefixes.index(prefix)
    filepath = filepath.relative_to(prefix)
    spectrogram_path = join('folder{}'.format(folder_index), str(filepath))
    return spectrogram_path


def writer_worker(writer):
    with initialization:
        initialization.wait_for(lambda: completed_inits.value == number_of_processes.value, timeout=15)
    print('Starting extraction with {} processes...'.format(number_of_processes.value))
    with tqdm(total=total_num_of_files) as pbar:
        poison_pills = 0
        while True:
            features = result_queue.get()
            if features:
                for index, file_name, feature_vector in features:
                    writer.write(feature_vector, file_name, index=index)
                result_queue.task_done()
                pbar.update(1)
            else:
                poison_pills += 1
                if poison_pills == number_of_processes.value:
                    result_queue.task_done()
                    break
                result_queue.task_done()


def kill_inactive(process_list):
    for process in process_list:
        if not process.is_alive():
            process.terminate()

if __name__ == '__main__':
    # set up the configuration object and parse commandline arguments
    configuration = Configuration()
    configuration.parse_arguments()

    # initialize feature writer and perform extraction
    feature_writer = FeatureWriter(configuration.output, configuration.label_dict)

    # initialize job and result queues
    job_queue = JoinableQueue()
    result_queue = JoinableQueue()
    total_num_of_files = len(configuration.files)
    number_of_processes = configuration.number_of_processes

    if not number_of_processes:
        number_of_processes = mp.cpu_count()

    number_of_processes = Value('i', number_of_processes)

    for f in configuration.files:
        job_queue.put(f)

    processes = []
    initialization = Condition()
    completed_inits = Value('i', 0)
    for i in range(number_of_processes.value):
        p = Process(target=extraction_worker,
                    args=(configuration, configuration.device_ids[i % len(configuration.device_ids)], i))
        p.daemon = True
        processes.append(p)
        p.start()

    writer_thread = Process(target=writer_worker, args=(feature_writer,))
    writer_thread.daemon = True
    writer_thread.start()

    with initialization:
        initialization.wait_for(lambda: completed_inits.value == number_of_processes.value, timeout=15)
        number_of_processes.value = sum(map(lambda x: x.is_alive(), processes))

    #kill dead processes
    kill_inactive(processes)

    # poison pilling for extraction workers
    for i in range(number_of_processes.value):
        job_queue.put(None)


    job_queue.join()
    result_queue.join()

    # extract(configuration, feature_writer)

