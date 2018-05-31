import numpy as np
import re
import gc
import tensorflow as tf
from collections import namedtuple

FeatureTuple = namedtuple('FeatureTuple', ['name', 'timestamp', 'features'])


class Extractor():
    def __init__(self, images, batch_size):
        self.images = _batch_images(images, batch_size)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return self

    def __next__(self):
        return self.extract_features(next(self.images))

    def extract_features(self, images):
        raise NotImplementedError(
            'Feature extractor must implement \'extract_features(self, images\' !'
        )


class TensorFlowExtractor(Extractor):

    def __init__(self, images, model_path, layer, batch_size=256, gpu=True):
        super().__init__(images, batch_size)
        self.batch_size = batch_size
        self.graph = self.__load_graph(model_path)
        self.layer = layer
        with self.graph.as_default():
            self.input, self.layers = self.__input_and_layers()
            assert self.layer in self.layers, '\'{}\' is not a valid layer. Available layers are: {}'.format(
                self.layer, sorted(list(self.layers.keys())))
            net_output = self.layers[self.layer]
            dim = tf.reduce_prod(tf.shape(net_output)[1:])
            self.features = tf.reshape(net_output, [-1, dim])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.layers = self.layers.keys()

    def __load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="extractor")
        return graph

    def __input_and_layers(self):
        tensor_names = [op.name + ':0' for op in self.graph.get_operations()]
        prefix_re = r'^(\w+)/*'
        prefix = re.match(prefix_re, tensor_names[0]).group(1)
        layer_re = r'\b(\w+)/\1\b'
        layer_names = set(
            re.search(layer_re, tensor).group(1) for tensor in tensor_names
            if re.search(layer_re, tensor))
        layer_dict = {
            layer: self.graph.get_tensor_by_name(
                '/'.join([prefix] + [layer] * 2) + ':0')
            for layer in layer_names
        }
        input = self.graph.get_tensor_by_name(prefix + '/input:0')
        return input, layer_dict

    def __process_images(self, images, data_spec):

        if data_spec.expects_bgr:
            # convert from RGB to BGR
            images = images[:, :, :, ::-1]
        # Rescale
        images = tf.image.resize_images(
            images, (data_spec.crop_size, data_spec.crop_size))
        return tf.to_float(images)

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        feature_batch = self.session.run(
            self.features, feed_dict={
                self.input: image_batch
            })
        return map(FeatureTuple._make, zip(name_batch, ts_batch,
                                           feature_batch))


def _batch_images(images, batch_size=256):
    current_name_batch = []
    current_ts_batch = []
    current_image_batch = []
    index = 0
    while True:
        try:
            name, ts, image = next(images)
            current_name_batch.append(name)
            current_ts_batch.append(ts)
            current_image_batch.append(image)
            del image
            if (index + 1) % batch_size == 0:
                name_batch, ts_batch, image_batch = current_name_batch, current_ts_batch, np.array(
                    current_image_batch, dtype=np.float32)
                current_name_batch = []
                current_ts_batch = []
                current_image_batch = []
                gc.collect()
                yield (name_batch, ts_batch, image_batch)
            index += 1
        except StopIteration:
            if current_name_batch:
                name_batch, ts_batch, image_batch = current_name_batch, current_ts_batch, np.array(
                    current_image_batch, dtype=np.float32)
                current_name_batch = []
                gc.collect()
                yield (name_batch, ts_batch, image_batch)
            else:
                gc.collect()
                raise StopIteration
