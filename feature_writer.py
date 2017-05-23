class FeatureWriter:
    """
    Class that handles writing the deep spectrum features to a file. Supports both csv and arff formats.
    """
    def __init__(self, output, label_dict=None):
        # determine mode (arff/csv) and load labels
        self.arff_mode = output.endswith('.arff')
        self.write_header = True
        self.output = output
        self.label_dict = label_dict
        if label_dict is None:
            print('No labels specified. Defaulting to \'?\'')
            self.labels = '?'
        else:
            self.labels = set(self.label_dict.values())

    def write(self, features, name, index=None):
        """
        Writes features to file.
        :param features: features to write
        :param name: name to use for the feature instance
        :param index: time series index of the features
        :return: Nothing
        """
        if self.arff_mode:
            self._write_arff(features, name, index)
        else:
            self._write_csv(features, name, index)

    def _write_csv(self, features, name, index=None):
        """
        Writes features to csv.
        :param features: features to write
        :param name: name to use for the feature instance
        :param index: time series index of the features
        :return: Nothing
        """
        label = self.label_dict[name] if self.label_dict is not None else '?'
        if self.write_header:
            self._write_csv_header(features, index)
        with open(self.output, 'a') as f:
            if index is None:
                line = ','.join(map(str, [name] + features + [label]))
            else:
                line = ','.join(map(str, [name] + [index] + features + [label]))
            f.write(line + '\n')

    def _write_csv_header(self, features, index):
        """
        Writes a csv header that suits the given features.
        :param features: features used for determining the number of attributes in the header
        :param index: if given, an index attribute is added to the header
        :return: Nothing
        """
        if index is None:
            header = ','.join(['name'] + ['neuron_' + str(i) for i in range(len(features))] + ['class'])
        else:
            header = ','.join(['name'] + ['index'] + ['neuron_' + str(i) for i in range(len(features))] + ['class'])
        with open(self.output, 'w') as f:
            f.write(header + '\n')
        self.write_header = False

    def _write_arff(self, features, name, index=None):
        """
        Writes features to arff.
        :param features: features to write
        :param name: name to use for the feature instance
        :param index: time series index of the features
        :return: Nothing
        """
        label = self.label_dict[name] if self.label_dict is not None else '?'
        if self.write_header:
            self._write_arff_header(features, index)
        with open(self.output, 'a') as f:
            if index is None:
                line = ','.join(map(str, [name] + features + [label]))
            else:
                line = ','.join(map(str, [name] + [index] + features + [label]))
            f.write(line + '\n')

    def _write_arff_header(self, features, index):
        """
        Writes an arff header that suits the given features.
        :param features: features used for determining the number of attributes in the header
        :param index: if given, an index attribute is added to the header
        :return: Nothing
        """
        with open(self.output, 'w') as f:
            f.write('@relation \'Deep Spectrum Features\'\n\n@attribute name string\n')
            if index is not None:
                f.write('@attribute index numeric\n')
            for i, feature in enumerate(features):
                f.write('@attribute neuron_' + str(i) + ' numeric\n')
            f.write('@attribute class {' + ','.join(self.labels) + '}\n\n')
            f.write('@data\n')
        self.write_header = False
