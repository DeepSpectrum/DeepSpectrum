import argparse
import csv
from os import makedirs
from os.path import dirname, splitext, abspath

import arff
import numpy as np
import pandas as pd
from tqdm import tqdm

import custom_arff


def _load_features(feature_file):
    """
    Loads a feature file into a pandas dataframe
    :param feature_file: the file to load
    :return: dataframe holding the features
    """
    if feature_file.endswith('.arff'):
        arff_data = arff.load(open(feature_file))
        header = [attribute[0] for attribute in arff_data['attributes']]
        return pd.DataFrame(arff_data['data'], columns=header), arff_data['attributes'], arff_data['relation']
    else:
        return pd.read_csv(feature_file), None, None


def reduce_features(input_file, output_file):
    """
    Reduces a feauture file by removing all-zero attributes.
    :param input_file: the file to reduce
    :param output_file: where to write the reduced file to
    :return: Nothing
    """
    print('Parsing feature file...')
    features, arff_header, arff_relation = _load_features(input_file)
    print('Reducing features...')
    reduced_features = pd.DataFrame()
    for column_name in tqdm(features):
        if not features[column_name].dtype == np.float64 or np.any(features[column_name].values):
            reduced_features[column_name] = features[column_name].values
    features = reduced_features
    print('Reduced feature size: ' + str(len(features.columns)))
    print('Writing output...')
    _write_features(features, output_file, arff_header, arff_relation)


def _write_features(features, output, arff_header, arff_relation):
    """
    Writes features to csv or arff
    :param features: the features to write
    :param output: output file path
    :param arff_header: in case of writing arff, the header can be specified here
    :param arff_relation: name of the output arff relation
    :return: nothing
    """
    makedirs(dirname(output), exist_ok=True)
    if output.endswith('.arff'):
        _write_arff(features, output, arff_header, arff_relation)
    else:
        _write_csv(features, output)


def _write_arff(features, output, arff_header, arff_relation):
    """
    Writes features to an arff file
    :param features: features to write
    :param output: output file path
    :param arff_header: arff header
    :param arff_relation: name of arff relation
    :return: Nothing
    """
    arff_data = dict()
    arff_data['relation'] = arff_relation + " reduced"
    arff_data['attributes'] = [attribute for attribute in arff_header if attribute[0] in features.columns]
    arff_data['data'] = features.values
    arff.dump(arff_data, open(output, 'w'))


def _write_csv(features, output):
    """
    Writes features to a csv file
    :param features: features to write
    :param output: output file path
    :return: Nothing
    """
    features.to_csv(output)


def _reduce(file_name):
    with open(file_name, newline='') as file:
        if file_name.endswith('.arff'):
            reader = custom_arff.Reader(file)
        else:
            reader = csv.reader(file, delimiter=',')
            next(reader)

        first_row = np.array(next(reader))
        indices_to_remove = set(np.where((first_row == '0.0') | (first_row == '0'))[0])
        for line in reader:
            data = np.array(line)
            indices_to_remove = indices_to_remove.intersection(np.where((data == '0.0') | (data == '0'))[0])
    return indices_to_remove


def _apply_reduction(input, output, indices_to_remove):
    with open(input, newline='') as input_file:
        with open(output, 'w', newline='') as output_file:
            input_arff = input.endswith('.arff')
            output_arff = output.endswith('.arff')
            if input_arff:
                reader = custom_arff.Reader(input_file)
                reduced_attributes = [reader.attributes[x] for x in range(len(reader.attributes)) if
                                      x not in indices_to_remove]
                if output_arff:
                    writer = custom_arff.Writer(output_file, '\'Reduced ' + reader.relation[1:], reduced_attributes)
                else:
                    writer = csv.writer(output_file, delimiter=',')
                    writer.writerow([attribute[0] for attribute in reduced_attributes])

            else:
                reader = csv.reader(input_file, delimiter=',')
                attributes = next(reader)
                reduced_attributes = [attributes[x] for x in range(len(attributes)) if x not in indices_to_remove]
                if output_arff:
                    arff_attributes = [(attribute_name, 'numeric') for attribute_name in reduced_attributes]
                    arff_attributes[0][1] = 'string'
                    arff_attributes[:-1][1] = 'nominal'
                    writer = custom_arff.Writer(output_file, '\'Reduced Features\'', arff_attributes)
                else:
                    writer = csv.writer(output_file, delimiter=',')
                    writer.writerow(reduced_attributes)

            for line in reader:
                reduced_line = [line[index] for index in range(len(line)) if index not in indices_to_remove]
                writer.writerow(reduced_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reduce a list of feature files by removing features that are always zero in the first given file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', nargs='+', help='The feature files to reduce.')
    args = vars(parser.parse_args())
    input_files = list(map(abspath, args['input']))
    print('Selecting features to remove from {}...'.format(input_files[0]))
    features_to_remove = _reduce(input_files[0])
    print('Removing {} features at the following indices: {}'.format(
        str(len(features_to_remove)), ','.join(list(map(str, sorted(features_to_remove))))))
    for input_file in input_files:
        output_file = splitext(input_file)[0] + '.reduced' + splitext(input_file)[1]
        print('Reducing {}...'.format(input_file))
        _apply_reduction(input_file, output_file, features_to_remove)
