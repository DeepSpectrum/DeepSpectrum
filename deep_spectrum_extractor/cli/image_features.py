import cv2
import deep_spectrum_extractor.tools.feature_reduction as fr
import numpy as np
from .configuration import Configuration
from ..tools.feature_writer import get_writer
from os import environ

environ['GLOG_minloglevel'] = '2'
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def image_reader(files):
    for image in files:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = img[:,:,::-1]
        yield np.array([img])

def main():
    configuration = Configuration(plotting=False, file_type='png')
    configuration.parse_arguments()
    plots = image_reader(configuration.files)
    print('Loading model and weights...')
    if configuration.backend == 'caffe':
        from deep_spectrum_extractor.backend.extractor import CaffeExtractor
        extractor = CaffeExtractor(
            images=plots, **configuration.extraction_args)
    elif configuration.backend == 'tensorflow':
        from deep_spectrum_extractor.backend.extractor import TensorFlowExtractor
        extractor = TensorFlowExtractor(
            images=plots, **configuration.extraction_args)

    if configuration.extraction_args['layer'] not in extractor.layers:
        configuration.parser.error(
            '\'{}\' is not a valid layer name for {}. Available layers are: {}'.
            format(configuration.extraction_args['layer'], configuration.net,
                   extractor.layers))

    print('Extracting features from images...')
    writer = get_writer(**configuration.writer_args)
    writer.write_features(configuration.files, extractor)

    if configuration.reduced:
        print('Performing feature reduction...')
        fr.reduce_file(configuration.writer_args['output'],
                       configuration.reduced)
if __name__=='__main__':
    main()
