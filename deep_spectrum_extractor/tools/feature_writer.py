import csv

import numpy as np
from os.path import basename
from tqdm import tqdm

from .custom_arff import ArffWriter


class FeatureWriter:
    def __init__(self, output, label_dict, labels, continuous_labels, window,
                 hop, start, no_timestamps, no_labels):
        self.output = output
        self.label_dict = label_dict
        self.labels = labels
        self.continuous_labels = continuous_labels
        self.window = window
        self.hop = hop
        self.start = start
        self.no_timestamps = no_timestamps
        self.no_labels = no_labels

    def write_features(self, names, features, hide_progress=False):
        raise NotImplementedError('write_features must be implemented!')

    def timestamp_and_label(self, file_name, idx):
        write_timestamp = self.window and not self.no_timestamps
        if write_timestamp:
            timestamp = self.start + (idx * self.hop)
            labels = self.label_dict[file_name][write_timestamp] if self.continuous_labels else \
                self.label_dict[file_name]
            return timestamp, labels
        else:
            return None, self.label_dict[file_name]


class ArffFeatureWriter(FeatureWriter):
    def write_features(self, names, features, hide_progress=False):
        write_timestamp = self.window and not self.no_timestamps
        with open(self.output, 'w', newline='') as output_file:
            writer = None
            for file_name, features in tqdm(
                    zip(names, features),
                    total=len(features),
                    disable=hide_progress):
                if self.no_labels:
                    classes = None
                else:
                    classes = [(class_name, '{' + ','.join(class_type) + '}')
                               if class_type else (class_name, 'numeric')
                               for class_name, class_type in self.labels]

                file_name = basename(file_name)
                for idx, feature_vector in enumerate(features):
                    if not writer:
                        attributes = _determine_attributes(
                            write_timestamp, feature_vector, classes)
                        writer = ArffWriter(
                            output_file, 'Deep Spectrum Features', attributes)
                    time_stamp, label = self.timestamp_and_label(
                        file_name, idx)
                    row = [file_name]
                    if time_stamp:
                        row.append(str(time_stamp))
                    row += (list(map(str, feature_vector)))
                    if not self.no_labels:
                        row += label
                    writer.writerow(row)


class CsvFeatureWriter(FeatureWriter):
    def write_features(self, names, features, hide_progress=False):
        write_timestamp = self.window and not self.no_timestamps
        with open(self.output, 'w', newline='') as output_file:
            writer = None
            for file_name, features in tqdm(
                    zip(names, features),
                    total=len(features),
                    disable=hide_progress):
                if self.no_labels:
                    classes = None
                else:
                    classes = [(class_name, '{' + ','.join(class_type) + '}')
                               if class_type else (class_name, 'numeric')
                               for class_name, class_type in self.labels]

                file_name = basename(file_name)
                for idx, feature_vector in enumerate(features):
                    if not writer:
                        attributes = _determine_attributes(
                            write_timestamp, feature_vector, classes)
                        writer = csv.writer(output_file, delimiter=',')
                        writer.writerow(
                            [attribute[0] for attribute in attributes])
                    time_stamp, label = self.timestamp_and_label(
                        file_name, idx)
                    row = [file_name]
                    if time_stamp:
                        row.append(time_stamp)
                    row += (list(map(str, feature_vector)))
                    if not self.no_labels:
                        row += label
                    writer.writerow(row)


class NumpyFeatureWriter(FeatureWriter):
    def write_features(self, names, features, hide_progress=False):
        file_names, timestamps, data, labels = ([], [], [], [])
        for file_name, features in tqdm(
                zip(names, features), total=len(features),
                disable=hide_progress):

            file_name = basename(file_name)

            for idx, feature_vector in enumerate(features):
                timestamp, label = self.timestamp_and_label(file_name, idx)
                if len(label) == 1:
                    label = label[0]
                data.append(feature_vector)
                labels.append(label)
                file_names.append(file_name)
                timestamps.append(timestamp)
        print('Writing output...')
        np.savez(
            self.output,
            names=file_names,
            features=data,
            labels=labels,
            timestamps=timestamps)


class TfFeatureWriter(FeatureWriter):
    try:
        import tensorflow as tf
    except:
        pass

    def write_features(self, names, features, hide_progress=False):
        label_map = [{
            label: index
            for index, label in enumerate(sorted(class_name[1]))
        } for class_name in self.labels]
        with self.tf.python_io.TFRecordWriter(self.output) as writer:
            for file_name, features in tqdm(
                    zip(names, features),
                    total=len(features),
                    disable=hide_progress):
                file_name = basename(file_name)
                for idx, feature_vector in enumerate(features):
                    time_stamp, label = self.timestamp_and_label(
                        file_name, idx)
                    label = [label_map[i][l] for i, l in enumerate(label)]
                    feature = {
                        'label': self._int64_feature(label),
                        'features': self._bytes_feature(
                            feature_vector.tostring())
                    }
                    example = self.tf.train.Example(
                        features=self.tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

    def _int64_feature(self, value):
        return self.tf.train.Feature(
            int64_list=self.tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return self.tf.train.Feature(
            bytes_list=self.tf.train.BytesList(value=[value]))

    def _floats_feature(self, value):
        return self.tf.train.Feature(
            float_list=self.tf.train.FloatList(value=value))


def _determine_attributes(timestamp, feature_vector, classes):
    if timestamp:
        attributes = [('name', 'string'), ('timeStamp', 'numeric')
                      ] + [('neuron_' + str(i), 'numeric')
                           for i, _ in enumerate(feature_vector)]
    else:
        attributes = [('name', 'string')
                      ] + [('neuron_' + str(i), 'numeric')
                           for i, _ in enumerate(feature_vector)]
    if classes:
        attributes += classes
    return attributes


def get_writer(**kwargs):
    if kwargs['output'].endswith('.arff'):
        return ArffFeatureWriter(**kwargs)
    elif kwargs['output'].endswith('.csv'):
        return CsvFeatureWriter(**kwargs)
    elif kwargs['output'].endswith('.npz'):
        return NumpyFeatureWriter(**kwargs)
    elif kwargs['output'].endswith('.tfrecord'):
        return TfFeatureWriter(**kwargs)
