import logging

log = logging.getLogger(__name__)


class ArffWriter():
    def __init__(self, file_object, relation_name, attributes):
        self.arff_file = file_object
        self.relation_name = relation_name
        self.attributes = attributes
        self._write_header()

    def _write_header(self):
        self.arff_file.write(' '.join(
            ['@relation', '\'{}\''.format(self.relation_name) + '\n']))
        self.arff_file.write('\n')
        for attribute_name, attribute_type in self.attributes:
            self.arff_file.write(
                ' '.join(['@attribute', attribute_name, attribute_type]) +
                '\n')
        self.arff_file.write('\n')
        self.arff_file.write('@data\n')

    def writerow(self, row):
        self.arff_file.write(','.join(row) + '\n')
