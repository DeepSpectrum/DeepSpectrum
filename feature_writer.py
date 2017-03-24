class FeatureWriter:
    def __init__(self, output, label_dict=None):
        self.arff_mode = output.endswith('.arff')
        self.write_header = True
        self.output = output
        self.label_dict = label_dict
        if label_dict is None:
            print('No labels specified. Defaulting to \'?\'')
            self.labels = '?'
        else:
            self.labels = self.label_dict

    def write(self, features, name, index=None):
        if self.arff_mode:
            self._write_arff(features, name, index)
        else:
            self._write_csv(features, name, index)

    def _write_csv(self, features, name, index=None):
        label = self.label_dict[name] if self.label_dict is not None else '?'
        if self.write_header:
            self._write_csv_header(features, index)
        with open(self.output, 'a') as f:
            if index is None:
                line = ','.join(map(str, [name] + features.tolist() + [label]))
            else:
                line = ','.join(map(str, [name] + [index] + features.tolist() + [label]))
            f.write(line + '\n')

    def _write_csv_header(self, features, index):
        if index is None:
            header = ','.join(['name'] + ['neuron_' + str(i) for i in range(len(features))] + ['class'])
        else:
            header = ','.join(['name'] + ['index'] + ['neuron_' + str(i) for i in range(len(features))] + ['class'])
        with open(self.output, 'w') as f:
            f.write(header + '\n')
        self.write_header = False

    def _write_arff(self, features, name, index=None):
        label = self.label_dict[name] if self.label_dict is not None else '?'
        if self.write_header:
            self._write_arff_header(features, index)
        with open(self.output, 'a') as f:
            if index is None:
                line = ','.join(map(str, [name] + features.tolist() + [label]))
            else:
                line = ','.join(map(str, [name] + [index] + features.tolist() + [label]))
            f.write(line + '\n')

    def _write_arff_header(self, features, index):
        with open(self.output, 'w') as f:
            f.write('@RELATION \'Deep Spectrum Features\'\n\n@ATTRIBUTE name string\n')
            if index is not None:
                f.write('@ATTRIBUTE index numeric\n')
            for i, feature in enumerate(features):
                f.write('@ATTRIBUTE neuron_' + str(i) + ' numeric\n')
            f.write('@ATTRIBUTE class {' + ','.join(self.labels) + '}\n\n')
            f.write('@DATA\n')
        self.write_header = False
