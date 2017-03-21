import arff
import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_features(feature_file):
    if feature_file.endswith('.arff'):
        arff_data = arff.load(open(feature_file))
        header = [attribute[0] for attribute in arff_data['attributes']]
        return pd.DataFrame(arff_data['data'], columns=header), arff_data['attributes'], arff_data['relation']
    else:
        return pd.read_csv(feature_file), None, None


def reduce_features(input_file, output_file):
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
    if output.endswith('.arff'):
        _write_arff(features, output, arff_header, arff_relation)
    else:
        _write_csv(features, output)


def _write_arff(features, output, arff_header, arff_relation):
    arff_data = {}
    arff_data['relation'] = arff_relation+" reduced"
    arff_data['attributes'] = [attribute for attribute in arff_header if attribute[0] in features.columns]
    arff_data['data'] = features.values
    arff.dump(arff_data, open(output, 'w'))


def _write_csv(features, output):
    features.to_csv(output)


if __name__=='__main__':
