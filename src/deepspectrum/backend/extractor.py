import gc
from collections import namedtuple

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import torch
from torchvision import models, transforms
from PIL import Image

import logging

tf.compat.v1.logging.set_verbosity(logging.ERROR)

log = logging.getLogger(__name__)
tf.compat.v1.keras.backend.clear_session()

log.debug(f'Collected garbage {gc.collect()}') # if it's done something you should see a number being outputted

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

FeatureTuple = namedtuple("FeatureTuple", ["name", "timestamp", "features"])

eps = 1e-8


def mask(func):
    def mask_loss_function(*args, **kwargs):
        mask = tf.cast(tf.not_equal(tf.sign(args[0]), -1), tf.float32) + eps
        return func(args[0] * mask, args[1] * mask)

    return mask_loss_function


class Extractor:
    def __init__(self, images, batch_size):
        self.batch_size = batch_size
        self.set_images(images)
        
    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.extract_features(next(self.images))
        except StopIteration:
            raise StopIteration

    def set_images(self, images):
        self.images = _batch_images(images, batch_size=self.batch_size)
        
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

    @staticmethod
    def __preprocess_default(x):
        x = x.astype(np.float32)
        x /= 127.5
        x -= 1.
        return x

    def __init__(self,
                 images,
                 model_key,
                 layer,
                 weights_path="imagenet",
                 batch_size=256):
        super().__init__(images, batch_size)
        # reset_keras() 
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
        self.layer = layer
        if model_key in self.models:
            base_model = self.models[model_key](weights=weights_path)
            self.preprocess = self.preprocessors[model_key]
        else:
            log.info(
                f'{model_key} not available in Keras Applications. Trying to load model file from {weights_path}.'
            )
            base_model = tf.keras.models.load_model(
                weights_path,
                custom_objects={
                    'mask_loss_function':
                    mask(tf.keras.losses.categorical_crossentropy)
                })
            self.preprocess = self.__preprocess_default
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
    def __preprocess_alexnet(x):
        preprocess = transforms.Compose(
            [transforms.Resize(227),
             transforms.ToTensor()])
        x = torch.stack(
            [preprocess(Image.fromarray(image, mode="RGB")) for image in x])
        return x

    @staticmethod
    def __preprocess_squeezenet(x):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(), normalize])
        x = torch.stack(
            [preprocess(Image.fromarray(image, mode="RGB")) for image in x])
        return x

    @staticmethod
    def __preprocess_googlenet(x):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(), normalize])
        x = torch.stack(
            [preprocess(Image.fromarray(image, mode="RGB")) for image in x])
        return x

    def __init__(self, images, model_key, layer, batch_size=256):
        super().__init__(images, batch_size)
        self.models = {
            "alexnet": models.alexnet,
            "squeezenet": models.squeezenet1_1,
            "googlenet": models.googlenet
        }
        self.preprocessors = {
            "alexnet": self.__preprocess_alexnet,
            "squeezenet": self.__preprocess_squeezenet,
            "googlenet": self.__preprocess_googlenet
        }
        self.layer = layer
        self.model_key = model_key

        self.model, self.feature_layer, self.output_size = self.__build_model(
            layer)

    def __build_model(self, layer):

        assert (self.model_key in self.models
                ), f"Invalid model for pytorch extractor. Available models: \
            {self.models}"

        base_model = self.models[self.model_key](pretrained=True)
        base_model.eval()
        if self.model_key == "alexnet":
            log.debug(f'Layout of base model: \n{base_model}')
            layers = {"fc6": -5, "fc7": -2}
            assert (layer in layers
                    ), f"Invalid layer key. Available layers: {layers.keys}"

            feature_layer = base_model.classifier[layers[layer]]
            return base_model, feature_layer, (4096, )
        elif self.model_key == "squeezenet":
            log.info(
                f'Disregarding user choice of feature layer: Only one layer is currently available for squeezenet.'
            )
            base_model = torch.nn.Sequential(
                base_model.features,
                torch.nn.AdaptiveAvgPool2d(output_size=(2, 2)))
            feature_layer = base_model[-1]
            log.debug(f'Layout of model: \n{base_model}')

            return base_model, feature_layer, (512, 2, 2)

        elif self.model_key == "googlenet":
            layers = {"avgpool": base_model.avgpool, "fc": base_model.fc}
            assert (layer in layers
                    ), f"Invalid layer key. Available layers: {layers.keys}"
            feature_layer = layers[layer]
            log.debug(f'Layout of model: \n{base_model}')
            return base_model, feature_layer, (1024, 1, 1)

        else:
            pass

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        image_batch = self.preprocessors[self.model_key](image_batch)

        feature_vec = torch.zeros(image_batch.shape[0], *self.output_size)

        def copy_data(m, i, o):
            feature_vec.copy_(o.data)

        hook = self.feature_layer.register_forward_hook(copy_data)
        _ = self.model(image_batch)
        hook.remove()

        feature_batch = feature_vec.numpy()
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
