import arff
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    arff_data['relation'] = arff_relation+" reduced"
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


