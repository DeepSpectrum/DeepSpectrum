import gc
from collections import namedtuple

import numpy as np
import os
import tensorflow as tf
import torch
from torchvision import models
from PIL import Image
from collections import OrderedDict

import logging

log = logging.getLogger(__name__)


def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """

    # noinspection PyPackageRequirements

    tf.logging.set_verbosity(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


tensorflow_shutup()

FeatureTuple = namedtuple("FeatureTuple", ["name", "timestamp", "features"])


class Extractor:
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
            """Feature extractor must implement 'extract_features(self, images'\
                 !""")


class KerasExtractor(Extractor):
    @staticmethod
    def __resize(x, target_size=(224, 224)):
        if (x.shape[1], x.shape[2]) != target_size:
            x = np.array([
                np.array(
                    Image.fromarray(image, mode="RGB").resize(target_size))
                for image in x
            ])
        return x

    @staticmethod
    def __preprocess_vgg(x):
        x = x[:, :, :, ::-1]
        return x

    def __init__(self,
                 images,
                 model_key,
                 layer,
                 weights_path="imagenet",
                 batch_size=256):
        super().__init__(images, batch_size)
        self.models = {
            "vgg16":
            tf.keras.applications.vgg16.VGG16,
            "vgg19":
            tf.keras.applications.vgg19.VGG19,
            "resnet50":
            tf.keras.applications.resnet50.ResNet50,
            "xception":
            tf.keras.applications.xception.Xception,
            "inception_v3":
            tf.keras.applications.inception_v3,
            "densenet121":
            tf.keras.applications.densenet.DenseNet121,
            "densenet169":
            tf.keras.applications.densenet.DenseNet169,
            "densenet201":
            tf.keras.applications.densenet.DenseNet201,
            "mobilenet":
            tf.keras.applications.mobilenet.MobileNet,
            "mobilenet_v2":
            tf.keras.applications.mobilenet_v2.MobileNetV2,
            "nasnet_large":
            tf.keras.applications.nasnet.NASNetLarge,
            "nasnet_mobile":
            tf.keras.applications.nasnet.NASNetMobile,
            "inception_resnet_v2":
            tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
        }
        self.preprocessors = {
            "vgg16":
            self.__preprocess_vgg,
            "vgg19":
            self.__preprocess_vgg,
            "resnet50":
            tf.keras.applications.resnet50.preprocess_input,
            "xception":
            tf.keras.applications.xception.preprocess_input,
            "inception_v3":
            tf.keras.applications.inception_v3,
            "densenet121":
            tf.keras.applications.densenet.preprocess_input,
            "densenet169":
            tf.keras.applications.densenet.preprocess_input,
            "densenet201":
            tf.keras.applications.densenet.preprocess_input,
            "mobilenet":
            tf.keras.applications.mobilenet.preprocess_input,
            "mobilenet_v2":
            tf.keras.applications.mobilenet_v2.preprocess_input,
            "nasnet_large":
            tf.keras.applications.nasnet.preprocess_input,
            "nasnet_mobile":
            tf.keras.applications.nasnet.preprocess_input,
            "inception_resnet_v2":
            tf.keras.applications.inception_resnet_v2.preprocess_input,
        }
        self.batch_size = batch_size
        self.layer = layer
        base_model = self.models[model_key](weights=weights_path)
        if log.getEffectiveLevel() < logging.INFO:
            base_model.summary()
        self.layers = [layer.name for layer in base_model.layers]
        assert (layer in self.layers
                ), f"Invalid layer key. Available layers: {self.layers}"
        inputs = base_model.input
        outputs = (base_model.get_layer(layer)
                   if not hasattr(base_model.get_layer(layer), "output") else
                   base_model.get_layer(layer).output)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.preprocess = self.preprocessors[model_key]

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        image_batch = self.__resize(image_batch,
                                    target_size=self.model.input.shape[1:-1])
        image_batch = self.preprocess(image_batch)
        feature_batch = self.model.predict(image_batch)
        dim = np.prod(feature_batch.shape[1:])
        feature_batch = np.reshape(feature_batch, [-1, dim])

        return map(FeatureTuple._make, zip(name_batch, ts_batch,
                                           feature_batch))


class PytorchExtractor(Extractor):
    @staticmethod
    def __resize(x, target_size=(227, 227)):
        if (x.shape[1], x.shape[2]) != target_size:
            x = np.array([
                np.array(
                    Image.fromarray(image, mode="RGB").resize(target_size))
                for image in x
            ])
        return x

    @staticmethod
    def __preprocess_alexnet(x):
        x = PytorchExtractor.__resize(x, target_size=(227, 227))
        x = x / 255.
        return x

    def __init__(self, images, model_key, layer, batch_size=256):
        super().__init__(images, batch_size)
        self.models = {"alexnet": models.alexnet}
        self.preprocessors = {"alexnet": self.__preprocess_alexnet}
        self.batch_size = batch_size
        self.layer = layer
        self.model_key = model_key

        self.__build_model(layer)

    def __build_model(self, layer):

        assert (
            self.model_key in self.models
        ), f"Invalid model for pytorch extractor. Available models: {self.models}"
        base_model = self.models[self.model_key](pretrained=True)
        base_model.eval()
        if self.model_key == "alexnet":
            log.debug(f'Layout of base model: \n{base_model}')
            layers = {"fc6": -5, "fc7": -2}
            assert (layer in layers
                    ), f"Invalid layer key. Available layers: {layers}"

            # taken from pytorch alexnet code
            self.features = list(base_model.children())[0]
            self.avgpool = list(base_model.children())[1]
            self.classifier = list(base_model.children())[2][:layers[layer]]
        else:
            pass

    def forward(self, x):
        assert (
            self.model_key in self.models
        ), f"Invalid model for pytorch extractor. Available models: {self.models}"
        if self.model_key == "alexnet":
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
            return x
        else:
            pass

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        image_batch = self.preprocessors[self.model_key](image_batch)
        image_batch = torch.from_numpy(np.rollaxis(image_batch, 3, 1)).float()
        feature_batch = self.forward(image_batch)
        feature_batch = feature_batch.detach().numpy()
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
            name_batch, ts_batch, image_batch = (
                current_name_batch,
                current_ts_batch,
                np.array(current_image_batch, dtype=np.uint8),
            )
            current_name_batch = []
            current_ts_batch = []
            current_image_batch = []
            gc.collect()
            yield (name_batch, ts_batch, image_batch)
        index += 1

    if current_name_batch:
        name_batch, ts_batch, image_batch = (
            current_name_batch,
            current_ts_batch,
            np.array(current_image_batch, dtype=np.uint8),
        )
        gc.collect()
        yield (name_batch, ts_batch, image_batch)
    else:
        gc.collect()
        return
