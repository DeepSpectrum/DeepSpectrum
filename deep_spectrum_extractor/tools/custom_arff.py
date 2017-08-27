class Reader():
    def __init__(self, file_object):
        self.arff_file = file_object
        self.attributes = []
        self._parse_header()

    def _parse_header(self):
        for line in self.arff_file:
            line = line.strip()
            if line.lower().startswith('@relation'):
                self.relation = ' '.join(line.split(' ')[1:])[1:-1]
            if line.lower().startswith('@attribute'):
                self.attributes.append((line.split(' ')[1], line.split(' ')[2]))
            if line.lower().startswith('@data'):
                break

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.arff_file).strip().split(',')
        if len(line) == len(self.attributes):
            return line
        else:
            return self.__next__()


class Writer():
    def __init__(self, file_object, relation_name, attributes):
        self.arff_file = file_object
        self.relation_name = relation_name
        self.attributes = attributes
        self._write_header()

    def _write_header(self):
        self.arff_file.write(' '.join(['@relation', '\'{}\''.format(self.relation_name)+'\n']))
        self.arff_file.write('\n')
        for attribute_name, attribute_type in self.attributes:
            self.arff_file.write(' '.join(['@attribute', attribute_name, attribute_type]) + '\n')
        self.arff_file.write('\n')
        self.arff_file.write('@data\n')

    def writerow(self, row):
        self.arff_file.write(','.join(row)+'\n')

