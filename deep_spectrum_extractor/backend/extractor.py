from functools import reduce
from operator import mul


class TensorFlowExtractor():
    try:
        import tensorflow as tf
        from deep_spectrum_extractor import tf_models
    except ImportError:
        pass

    def __init__(self, net_name, weights_path, layer, gpu=True):
        self.input, self.net = self.__load_model(net_name)
        self.layer = layer
        net_output = self.net.layers[self.layer]
        dim = self.tf.reduce_prod(self.tf.shape(net_output)[1:])
        self.features = self.tf.reshape(net_output, [-1, dim])
        config = self.tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = self.tf.Session(config=config)
        self.net.load(weights_path, self.session)
        self.layers = self.net.layers.keys()

    def __load_model(self, name):
        '''Creates and returns an instance of the model given its class name.
        The created model has a single placeholder node for feeding images.
        '''
        # Find the model class from its name
        all_models = self.tf_models.get_models()
        lut = {model.__name__: model for model in all_models}
        if name not in lut:
            print('Invalid model index. Options are:')
            # Display a list of valid model names
            for model in all_models:
                print('\t* {}'.format(model.__name__))
            return None
        NetClass = lut[name]

        # Create a placeholder for the input image
        data_spec = self.tf_models.get_data_spec(model_class=NetClass)
        input = self.tf.placeholder(self.tf.float32,
                                         shape=(None, data_spec.crop_size, data_spec.crop_size, data_spec.channels))
        processed_images = self.__process_images(input, data_spec)
        # Construct and return the model
        return input, NetClass({'data': processed_images})

    def __process_images(self, images, data_spec):

        if data_spec.expects_bgr:
            # convert from RGB to BGR
            images = images[:, :, :, ::-1]
        # Rescale
        images = self.tf.image.resize_images(images, (data_spec.crop_size, data_spec.crop_size))
        return self.tf.to_float(images)

    def extract_features(self, images):
        return self.session.run(self.features, feed_dict={self.input: images})


class CaffeExtractor():
    try:
        import caffe
    except ImportError:
        pass
    import numpy as np
    def __init__(self, def_path, weights_path, layer, gpu=True):
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




    def extract_features(self, images):
        # reshape input layer as batch processing is not needed
        shape = self.net.blobs['data'].shape
        self.net.blobs['data'].reshape(images.shape[0], shape[1], shape[2], shape[3])
        self.net.reshape()
        images = list(map(lambda x: self.transformer.preprocess('data', x), images))
        self.net.blobs['data'].data[...] = images
        self.net.forward()

        # extract features from the specified layer
        features = self.net.blobs[self.layer].data

        return self.np.reshape(features, (features.shape[0], -1))



