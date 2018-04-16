import numpy as np
import re

class Extractor():
    def __init__(self, images, batch_size):
        self.images = _new_batch_images(images, batch_size)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return self

    def __next__(self):
        return self.extract_features(next(self.images))

    def extract_features(self, images):
        raise NotImplementedError('Feature extractor must implement \'extract_features(self, images\' !')

class TensorFlowExtractor(Extractor):

    try:
        import tensorflow as tf
        from deep_spectrum_extractor import tf_models
    except ImportError:
        pass

    def __init__(self, images, model_path, layer, batch_size=256, gpu=True):
        super().__init__(images, batch_size)
        self.batch_size = batch_size
        self.graph = self.__load_graph(model_path)
        self.layer = layer
        with self.graph.as_default() as g:
            self.input, self.layers = self.__input_and_layers()
            assert self.layer in self.layers, '\'{}\' is not a valid layer. Available layers are: {}'.format(
                self.layer, sorted(list(self.layers.keys())))
            net_output = self.layers[self.layer]
            dim = self.tf.reduce_prod(self.tf.shape(net_output)[1:])
            self.features = self.tf.reshape(net_output, [-1, dim])
        config = self.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = self.tf.Session(graph=self.graph, config=config)
        self.layers = self.layers.keys()

    def __load_graph(self, frozen_graph_filename):
        with self.tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = self.tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name="extractor")
        return graph

    def __input_and_layers(self):
        tensor_names = [op.name+':0' for op in self.graph.get_operations()]
        prefix_re = r'^(\w+)/*'
        prefix = re.match(prefix_re, tensor_names[0]).group(1)
        layer_re = r'\b(\w+)/\1\b'
        layer_names = set(re.search(layer_re, tensor).group(1) for tensor in tensor_names if re.search(layer_re, tensor))
        layer_dict = {layer: self.graph.get_tensor_by_name('/'.join([prefix]+[layer]*2)+':0') for layer in layer_names}
        input = self.graph.get_tensor_by_name(prefix+'/input:0')
        return input, layer_dict

    def __process_images(self, images, data_spec):

        if data_spec.expects_bgr:
            # convert from RGB to BGR
            images = images[:, :, :, ::-1]
        # Rescale
        images = self.tf.image.resize_images(images, (data_spec.crop_size, data_spec.crop_size))
        return self.tf.to_float(images)

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        feature_batch = self.session.run(self.features, feed_dict={self.input: image_batch})
        return list(zip(name_batch, ts_batch, feature_batch))


class OldTensorFlowExtractor(Extractor):

    try:
        import tensorflow as tf
        from deep_spectrum_extractor import tf_models
    except ImportError:
        pass

    def __init__(self, images, model_path, layer, batch_size=256, gpu=True):
        super().__init__(images, batch_size)
        self.graph = self.__load_graph(model_path)
        self.layer = layer
        with self.graph.as_default() as g:
            self.input, self.layers = self.__input_and_layers()
            assert self.layer in self.layers, '\'{}\' is not a valid layer. Available layers are: {}'.format(
                self.layer, sorted(list(self.layers.keys())))
            net_output = self.layers[self.layer]
            dim = self.tf.reduce_prod(self.tf.shape(net_output)[1:])
            self.features = self.tf.reshape(net_output, [-1, dim])
        config = self.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = self.tf.Session(graph=self.graph, config=config)
        self.layers = self.layers.keys()

    def __load_graph(self, frozen_graph_filename):
        with self.tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = self.tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name="extractor")
        return graph

    def __input_and_layers(self):
        tensor_names = [op.name+':0' for op in self.graph.get_operations()]
        prefix_re = r'^(\w+)/*'
        prefix = re.match(prefix_re, tensor_names[0]).group(1)
        layer_re = r'\b(\w+)/\1\b'
        layer_names = set(re.search(layer_re, tensor).group(1) for tensor in tensor_names if re.search(layer_re, tensor))
        layer_dict = {layer: self.graph.get_tensor_by_name('/'.join([prefix]+[layer]*2)+':0') for layer in layer_names}
        input = self.graph.get_tensor_by_name(prefix+'/input:0')
        return input, layer_dict

    def __process_images(self, images, data_spec):

        if data_spec.expects_bgr:
            # convert from RGB to BGR
            images = images[:, :, :, ::-1]
        # Rescale
        images = self.tf.image.resize_images(images, (data_spec.crop_size, data_spec.crop_size))
        return self.tf.to_float(images)

    def extract_features(self, images):
        image_batches = batch_images(images, self.batch_size)
        all_features = []
        for images in image_batches:
            features = self.session.run(self.features, feed_dict={self.input: images})
            all_features.append(np.array(features))

        all_features = np.concatenate(all_features)
        return all_features


class CaffeExtractor(Extractor):
    try:
        import caffe
    except ImportError:
        pass

    def __init__(self, images, def_path, weights_path, layer, batch_size=256, gpu=True):
        super().__init__(images, batch_size)
        self.layer = layer
        # set mode to GPU or CPU computation
        if gpu:
            self.caffe.set_device(0)
            self.caffe.set_mode_gpu()
        else:
            self.caffe.set_mode_cpu()

        self.net = self.caffe.Net(def_path, weights_path, self.caffe.TEST)
        self.transformer = self.caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
        self.layers = list(self.net.blobs.keys())

    def extract_features(self, tuple_batch):
        name_batch, ts_batch, image_batch = tuple_batch
        shape = self.net.blobs['data'].shape
        self.net.blobs['data'].reshape(image_batch.shape[0], shape[1], shape[2], shape[3])
        self.net.reshape()
        image_batch = list(map(lambda x: self.transformer.preprocess('data', x), image_batch))
        self.net.blobs['data'].data[...] = image_batch
        self.net.forward()

        # extract features from the specified layer
        feature_batch = self.net.blobs[self.layer].data
        return list(zip(name_batch, ts_batch, feature_batch))

# class CaffeExtractor(Extractor):
#     try:
#         import caffe
#     except ImportError:
#         pass

#     def __init__(self, images, def_path, weights_path, layer, batch_size=256, gpu=True):
#         super().__init__(images)
#         self.batch_size = batch_size
#         self.layer = layer
#         # set mode to GPU or CPU computation
#         if gpu:
#             self.caffe.set_device(0)
#             self.caffe.set_mode_gpu()
#         else:
#             self.caffe.set_mode_cpu()

#         self.net = self.caffe.Net(def_path, weights_path, self.caffe.TEST)
#         self.transformer = self.caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
#         self.transformer.set_transpose('data', (2, 0, 1))
#         self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
#         self.layers = list(self.net.blobs.keys())




#     def extract_features(self, images):
#         image_batches = batch_images(images, self.batch_size)
#         all_features = []
#         for images in image_batches:
#             shape = self.net.blobs['data'].shape
#             self.net.blobs['data'].reshape(images.shape[0], shape[1], shape[2], shape[3])
#             self.net.reshape()
#             images = list(map(lambda x: self.transformer.preprocess('data', x), images))
#             self.net.blobs['data'].data[...] = images
#             self.net.forward()

#             # extract features from the specified layer
#             features = self.net.blobs[self.layer].data
#             all_features.append(np.array(features))

#         all_features = np.concatenate(all_features)
#         return np.reshape(all_features, (all_features.shape[0], -1))


def batch_images(images, batch_size=256):
    if images.shape[0] <= batch_size:
        return [images]
    else:
        return np.array_split(images, batch_size)


def _new_batch_images(images, batch_size=256):
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
            if (index + 1) % batch_size == 0:
                name_batch, ts_batch, image_batch = current_name_batch, current_ts_batch, np.array(current_image_batch, dtype=np.float32)
                current_name_batch = []
                current_ts_batch = []
                current_image_batch = []
                yield (name_batch, ts_batch, image_batch)
            index += 1
        except StopIteration:
            if current_name_batch:
                name_batch, ts_batch, image_batch = current_name_batch, current_ts_batch, np.array(current_image_batch, dtype=np.float32)
                current_name_batch = []
                yield (name_batch, ts_batch, image_batch)
            else:
                raise StopIteration
