import arff
import numpy as np
import tensorflow as tf
import pandas as pd
from functools import partial

_WEIGHT_COLUMN = 'weight'


class DataLoader():
    def __init__(self,
                 file_path,
                 sequences=True,
                 name_column=0,
                 label_column=-1,
                 timestamp_column=1,
                 regression=False,
                 max_sequence_len=None,
                 sequence_classification=True,
                 batch_size=128,
                 shuffle=False,
                 num_epochs=1,
                 num_threads=1,
                 queue_capacity=1000,
                 class_weights=None):
        self.sequences = sequences
        self.regression = regression
        self.name_column = name_column
        self.timestamp_column = timestamp_column
        self.label_column = label_column
        self.names = None
        self.num_features = None
        self.max_sequence_len = max_sequence_len
        self.sequence_classification = sequence_classification

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.num_threads = num_threads
        self.queue_capacity = queue_capacity
        self.steps_per_epoch = 0

        self.input_fn = None
        self.feature_columns = None
        self.weight_column = _WEIGHT_COLUMN
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.__infer_weights = False
        else:
            self.__infer_weights = True

        self.label_dict = dict()
        self.reverse_label_dict = dict()
        print('Loading data from {}...'.format(file_path))
        self.__parse_data(file_path)

    def __load_file(self, file_path):
        if file_path.endswith('.arff'):
            return self.__load_arff(file_path)
        elif file_path.endswith('.csv'):
            return self.__load_csv(file_path)

    def __load_arff(self, file_path):
        arff_data = arff.load(open(file_path))
        return np.array(arff_data['data'])

    def __load_csv(self, file_path):
        df = pd.read_csv(file_path, sep=';')
        return df.values

    def __parse_data(self, file_path):
        if file_path.endswith('.arff') or file_path.endswith('.csv'):
            self.__parse_numpy_data(file_path)

    def __parse_numpy_data(self, file_path):
        data = self.__load_file(file_path)
        num_columns = len(data[0])
        feature_indices = list(range(num_columns))
        labels = None
        if self.name_column is not None:
            feature_indices.remove(self.name_column)
            self.names = np.array(data[:, self.name_column], dtype=str)
        if self.timestamp_column is not None:
            feature_indices.remove(self.timestamp_column)
        if self.label_column is not None:
            self.label_column = num_columns + self.label_column if self.label_column < 0 else self.label_column
            feature_indices.remove(self.label_column)
            labels = self.__convert_labels(data[:, self.label_column])
            if not self.regression:
                weights = np.array(
                    list(map(lambda x: self.class_weights[x], labels)),
                    dtype=np.float32)
            else:
                weights = np.ones(labels.shape, dtype=np.float32)

        features = data[:, feature_indices].astype(np.float32)

        features, labels, weights = self.__pack_numpy_data(
            features, labels, weights)

        if not self.regression:
            print('Using the following class weights:')
            for label in self.class_weights:
                print('{}: {:.2f}'.format(label, self.class_weights[label]))
        self.feature_columns = [
            tf.feature_column.numeric_column(
                'features', shape=features[0].shape)
        ]
        self.weight_column = 'weight'
        self.steps_per_epoch = features.shape[0] / self.batch_size
        self.input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'features': features,
               self.weight_column: weights},
            y=labels,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            shuffle=self.shuffle,
            num_threads=self.num_threads,
            queue_capacity=self.queue_capacity)

    def __pack_numpy_data(self, features, labels, weights):
        if self.sequences:
            print('Packing sequences...')
            assert self.names is not None
            unique_names, indices = np.unique(self.names, return_index=True)
            sort_index = np.argsort(indices)
            self.names = unique_names[sort_index]
            sorted_indices = indices[sort_index]
            sequence_lens = np.ediff1d(sorted_indices + len(features))
            if self.max_sequence_len is None:
                self.max_sequence_len = max(sequence_lens)
            self.min_sequence_len = min(sequence_lens)
            self.mean_sequence_len = float(
                sum(sequence_lens) / max(len(sequence_lens), 1))
            features = np.split(features, sorted_indices[1:])
            features = np.array(
                tf.keras.preprocessing.sequence.pad_sequences(
                    features,
                    maxlen=self.max_sequence_len,
                    dtype='float32',
                    padding='post',
                    truncating='post'),
                dtype=np.float32)
            if labels is not None:
                labels = np.split(labels, sorted_indices[1:])
                labels = self.__pad_labels(labels)
                weights = np.split(weights, sorted_indices[1:])
                weights = np.array(
                    tf.keras.preprocessing.sequence.pad_sequences(
                        weights,
                        maxlen=self.max_sequence_len,
                        dtype='float32',
                        padding='post',
                        truncating='post'))
                if self.sequence_classification:
                    labels = np.array([labels[0] for labels in labels])
                    if self.__infer_weights:
                        unique, counts = np.unique(labels, return_counts=True)
                        self.class_weights = dict(
                            zip(unique, map(lambda x: max(counts)/x, counts)))
                        weights = np.array(
                            list(map(lambda x: self.class_weights[x], labels)),
                            dtype=np.float32)
                    else:
                        weights = np.array([weights[0] for weights in weights])
        return features, labels, weights

    def __pad_labels(self, labels):
        if self.regression:
            labels = np.array(
                tf.keras.preprocessing.sequence.pad_sequences(
                    labels,
                    maxlen=self.max_sequence_len,
                    dtype='float32',
                    padding='post',
                    truncating='post'),
                dtype=np.float32)
        else:
            labels = np.array(
                [
                    np.pad(
                        labels, (0, max(self.max_sequence_len - len(labels),0)),
                        mode='edge') if len(labels) < self.max_sequence_len else labels[:self.max_sequence_len] for labels in labels                 ],
                dtype=str)
        return labels

    def __convert_labels(self, labels):
        if self.regression:
            return np.array(labels, dtype=np.float32)
        else:
            self.label_dict = {
                label: index
                for index, label in enumerate(sorted(set(labels)))
            }
            self.reverse_label_dict = {
                key: value
                for key, value in zip(
                    sorted(self.label_dict.values()),
                    sorted(self.label_dict.keys()))
            }
            unique, counts = np.unique(labels, return_counts=True)

            if self.__infer_weights:
                self.class_weights = dict(
                    zip(unique, map(lambda x: max(counts)/x, counts)))
            return np.array(labels, dtype=str)
