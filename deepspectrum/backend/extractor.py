import gc
from collections import namedtuple

import numpy as np
from PIL import Image

import logging

log = logging.getLogger(__name__)

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation tools to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func

            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass


tensorflow_shutup()

FeatureTuple = namedtuple('FeatureTuple', ['name', 'timestamp', 'features'])


class Extractor():
    def __init__(self, images, batch_size):
        self.images = _batch_images(images, batch_size)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.extract_features(next(self.images))
        except StopIteration:
            raise StopIteration

    def extract_features(self, images):
        raise NotImplementedError(
            'Feature extractor must implement \'extract_features(self, images\' !'
        )


class KerasExtractor(Extractor):
    try:
        import tensorflow as tf
    except ImportError:
        pass

    @staticmethod
    def __resize(x, target_size=(224, 224)):
        if (x.shape[1], x.shape[2]) != target_size:
            x = np.array([np.array(Image.fromarray(image, mode='RGB').resize(target_size)) for image in x])
        return x

    @staticmethod
    def __preprocess_vgg(x):
        x = x[:, :, :, ::-1]
        return x

    def __init__(self, images, model_key, layer, weights_path='imagenet', batch_size=256):
        super().__init__(images, batch_size)
        self.models = {'vgg16': self.tf.keras.applications.vgg16.VGG16, 'vgg19': self.tf.keras.applications.vgg19.VGG19,
                       'resnet50': self.tf.keras.applications.resnet50.ResNet50,
                       'xception': self.tf.keras.applications.xception.Xception,
                       'inception_v3': self.tf.keras.applications.inception_v3,
                       'densenet121': self.tf.keras.applications.densenet.DenseNet121,
                       'densenet169': self.tf.keras.applications.densenet.DenseNet169,
                       'densenet201': self.tf.keras.applications.densenet.DenseNet201,
                       'mobilenet': self.tf.keras.applications.mobilenet.MobileNet,
                       'mobilenet_v2': self.tf.keras.applications.mobilenet_v2.MobileNetV2,
                       'nasnet_large': self.tf.keras.applications.nasnet.NASNetLarge,
                       'nasnet_mobile': self.tf.keras.applications.nasnet.NASNetMobile,
                       'inception_resnet_v2': self.tf.keras.applications.inception_resnet_v2.InceptionResNetV2}
        self.preprocessors = {'vgg16': self.__preprocess_vgg, 'vgg19': self.__preprocess_vgg,
                              'resnet50': self.tf.keras.applications.resnet50.preprocess_input,
                              'xception': self.tf.keras.applications.xception.preprocess_input,
                              'inception_v3': self.tf.keras.applications.inception_v3,
                              'densenet121': self.tf.keras.applications.densenet.preprocess_input,
                              'densenet169': self.tf.keras.applications.densenet.preprocess_input,
                              'densenet201': self.tf.keras.applications.densenet.preprocess_input,
                              'mobilenet': self.tf.keras.applications.mobilenet.preprocess_input,
                              'mobilenet_v2': self.tf.keras.applications.mobilenet_v2.preprocess_input,
                              'nasnet_large': self.tf.keras.applications.nasnet.preprocess_input,
                              'nasnet_mobile': self.tf.keras.applications.nasnet.preprocess_input,
                              'inception_resnet_v2': self.tf.keras.applications.inception_resnet_v2.preprocess_input}
        self.batch_size = batch_size
        self.layer = layer
        base_model = self.models[model_key](weights=weights_path)
        if log.getEffectiveLevel() < logging.INFO:
            base_model.summary()
        self.layers = [layer.name for layer in base_model.layers]
        assert layer in self.layers, f'Invalid layer key. Available layers: {self.layers}'
        inputs = base_model.input
        outputs = base_model.get_layer(layer) if not hasattr(base_model.get_layer(layer),
                                                             'output') else base_model.get_layer(layer).output
        self.model = self.tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.preprocess = self.preprocessors[model_key]

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        image_batch = self.__resize(image_batch, target_size=self.model.input.shape[1:-1])
        image_batch = self.preprocess(image_batch)
        feature_batch = self.model.predict(image_batch)
        dim = np.prod(feature_batch.shape[1:])
        feature_batch = np.reshape(feature_batch, [-1, dim])

        return map(FeatureTuple._make, zip(name_batch, ts_batch,
                                           feature_batch))


def _batch_images(images, batch_size=256):
    current_name_batch = []
    current_ts_batch = []
    current_image_batch = []
    index = 0
    for plot_tuple in images:

        name, ts, image = plot_tuple
        current_name_batch.append(name)
        current_ts_batch.append(ts)
        current_image_batch.append(image)
        del image
        if (index + 1) % batch_size == 0:
            name_batch, ts_batch, image_batch = current_name_batch, current_ts_batch, np.array(
                current_image_batch, dtype=np.uint8)
            current_name_batch = []
            current_ts_batch = []
            current_image_batch = []
            gc.collect()
            yield (name_batch, ts_batch, image_batch)
        index += 1

    if current_name_batch:
        name_batch, ts_batch, image_batch = current_name_batch, current_ts_batch, np.array(
            current_image_batch, dtype=np.uint8)
        gc.collect()
        yield (name_batch, ts_batch, image_batch)
    else:
        gc.collect()
        return
