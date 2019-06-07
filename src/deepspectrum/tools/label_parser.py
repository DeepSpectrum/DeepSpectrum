import csv
import decimal
from os.path import splitext, normpath


class LabelParser():
    def __init__(self,
                 filepath,
                 delimiter=',',
                 timecontinuous=False,
                 remove_extension=False):
        self._timecontinuous = timecontinuous
        self._filepath = filepath
        self._delimiter = delimiter
        self._remove_extension = remove_extension
        self.labels = []
        self.label_dict = {}

    def parse_labels(self):
        # delimiters are decided by the extension of the labels file

        reader = csv.reader(open(self._filepath, newline=''),
                            delimiter=self._delimiter)

        header = next(reader)
        first_class_index = 2 if self._timecontinuous else 1

        classes = header[first_class_index:]

        # a list of distinct labels is needed for deciding on the nominal class values for .arff files
        self.labels = [[class_name, []] for class_name in classes]

        # parse the label file line by line
        for row in reader:
            name = splitext(normpath(
                row[0]))[0] if self._remove_extension else normpath(row[0])
            if self._timecontinuous:
                if name not in self.label_dict:
                    self.label_dict[name] = {}
                self.label_dict[name][decimal.Decimal(
                    row[1])] = row[first_class_index:]
            else:
                self.label_dict[name] = row[first_class_index:]
            for i, label in enumerate(row[first_class_index:]):
                if self._is_number(label):
                    self.labels[i] = (self.labels[i][0], None)
                else:
                    self.labels[i][1].append(label)
                    self.labels[i] = [
                        self.labels[i][0],
                        sorted(list(set(self.labels[i][1])))
                    ]

    @staticmethod
    def _is_number(s):
        try:
            complex(s)  # for int, long, float and complex
        except ValueError:
            return False

        return True
