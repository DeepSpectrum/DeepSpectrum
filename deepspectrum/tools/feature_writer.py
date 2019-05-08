import csv

from tqdm import tqdm

from .custom_arff import ArffWriter

import logging

log = logging.getLogger(__name__)


class FeatureWriter:
    def __init__(self, output, label_dict, labels, continuous_labels,
                 write_timestamps, no_labels):
        self.output = output
        self.label_dict = label_dict
        self.labels = labels
        self.continuous_labels = continuous_labels
        self.no_labels = no_labels
        self.write_timestamps = write_timestamps

    def write_features(self, names, features, hide_progress=False):
        raise NotImplementedError('write_features must be implemented!')

    def timestamp_and_label(self, file_name, timestamp):
        if self.write_timestamps:
            labels = self.label_dict[file_name][timestamp] if self.continuous_labels else \
                self.label_dict[file_name]
            return timestamp, labels
        else:
            return None, self.label_dict[file_name]


class ArffFeatureWriter(FeatureWriter):
    def write_features(self, names, features, hide_progress=False):
        with open(self.output, 'w', newline='') as output_file, tqdm(
                total=len(names),
                disable=log.getEffectiveLevel() >= logging.ERROR) as pbar:
            writer = None
            first = True
            for batch in features:
                for feature_tuple in batch:
                    if first:
                        old_name = feature_tuple.name
                        first = False
                    if self.no_labels:
                        classes = None
                    else:
                        classes = [(class_name, '{' + ','.join(class_type) +
                                    '}') if class_type else
                                   (class_name, 'numeric')
                                   for class_name, class_type in self.labels]
                    if not writer:
                        attributes = _determine_attributes(
                            self.write_timestamps, feature_tuple.features,
                            classes)
                        writer = ArffWriter(output_file,
                                            'Deep Spectrum Features',
                                            attributes)
                    time_stamp, label = self.timestamp_and_label(
                        feature_tuple.name, feature_tuple.timestamp)
                    row = [feature_tuple.name]
                    if time_stamp is not None:
                        row.append(str(time_stamp))
                    row += (list(map(str, feature_tuple.features)))
                    if not self.no_labels:
                        row += label
                    writer.writerow(row)
                    if feature_tuple.name != old_name:
                        pbar.update()
                        old_name = feature_tuple.name
                    del feature_tuple
            pbar.update()


class CsvFeatureWriter(FeatureWriter):
    def write_features(self, names, features, hide_progress=False):
        with open(self.output, 'w', newline='') as output_file, tqdm(
                total=len(names),
                disable=log.getEffectiveLevel() >= logging.ERROR) as pbar:
            writer = None
            first = True
            for batch in features:
                for feature_tuple in batch:
                    if first:
                        old_name = feature_tuple.name
                        first = False
                    if self.no_labels:
                        classes = None
                    else:
                        classes = [(class_name, '{' + ','.join(class_type) +
                                    '}') if class_type else
                                   (class_name, 'numeric')
                                   for class_name, class_type in self.labels]

                    if not writer:
                        attributes = _determine_attributes(
                            self.write_timestamps, feature_tuple.features,
                            classes)
                        writer = csv.writer(output_file, delimiter=',')
                        writer.writerow(
                            [attribute[0] for attribute in attributes])
                    time_stamp, label = self.timestamp_and_label(
                        feature_tuple.name, feature_tuple.timestamp)
                    row = [feature_tuple.name]
                    if time_stamp is not None:
                        row.append(time_stamp)
                    row += (list(map(str, feature_tuple.features)))
                    if not self.no_labels:
                        row += label
                    writer.writerow(row)
                    if feature_tuple.name != old_name:
                        pbar.update()
                        old_name = feature_tuple.name
            pbar.update()


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
