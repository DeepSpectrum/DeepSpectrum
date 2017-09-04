import sys
import os.path as osp
import numpy as np
import tensorflow as tf

# Add the kaffe module to the import path
#sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), '../../../')))

#from googlenet import GoogleNet
#from vgg import VGG16
from deep_spectrum_extractor.models.alexnet import AlexNet
#from caffenet import CaffeNet
#from nin import NiN
#from resnet import ResNet50, ResNet101, ResNet152

from cv2 import resize

class DataSpec(object):
    '''Input data specifications for an ImageNet model.'''

    def __init__(self,
                 batch_size,
                 scale_size,
                 crop_size,
                 isotropic,
                 channels=3,
                 mean=None,
                 bgr=True):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The image should be scaled to this size first during preprocessing
        self.scale_size = scale_size
        # Whether the model expects the rescaling to be isotropic
        self.isotropic = isotropic
        # A square crop of this dimension is expected by this model
        self.crop_size = crop_size
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # The values below are ordered BGR, as many Caffe models are trained in this order.
        # Some of the earlier models (like AlexNet) used a spatial three-channeled mean.
        # However, using just the per-channel mean values instead doesn't affect things too much.
        self.mean = mean if mean is not None else np.array([104., 117., 124.])
        # Whether this model expects images to be in BGR order
        self.expects_bgr = True


def alexnet_spec(batch_size=500):
    '''Parameters used by AlexNet and its variants.'''
    return DataSpec(batch_size=batch_size, scale_size=256, crop_size=227, isotropic=False)


def std_spec(batch_size, isotropic=True):
    '''Parameters commonly used by "post-AlexNet" architectures.'''
    return DataSpec(batch_size=batch_size, scale_size=256, crop_size=224, isotropic=isotropic)

# Collection of sample auto-generated models
MODELS = (AlexNet,) #CaffeNet, GoogleNet, NiN, ResNet50, ResNet101, ResNet152, VGG16)

# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
# The recommended batch size is based on a Titan X (12GB).
MODEL_DATA_SPECS = {
    AlexNet: alexnet_spec(),
    #CaffeNet: alexnet_spec(),
    #GoogleNet: std_spec(batch_size=200, isotropic=False),
    #ResNet50: std_spec(batch_size=25),
    #ResNet101: std_spec(batch_size=25),
    #ResNet152: std_spec(batch_size=25),
    #NiN: std_spec(batch_size=500),
    #VGG16: std_spec(batch_size=25)
}


def get_models():
    '''Returns a tuple of sample models.'''
    return MODELS


def get_data_spec(model_instance=None, model_class=None):
    '''Returns the data specifications for the given network.'''
    model_class = model_class or model_instance.__class__
    return MODEL_DATA_SPECS[model_class]

def load_model(name):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    # Find the model class from its name
    all_models = get_models()
    lut = {model.__name__: model for model in all_models}
    if name not in lut:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = lut[name]

    # Create a placeholder for the input image
    spec = get_data_spec(model_class=NetClass)
    data_node = tf.placeholder(tf.float32,
                               shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct and return the model
    return NetClass({'data': data_node})

def process_image(img, data_spec):

    if data_spec.expects_bgr:
        # Convert from RGB channel ordering to BGR
        # This matches, for instance, how OpenCV orders the channels.
        img = img[:,:,::-1]
    # Rescale
    img = resize(img, (data_spec.crop_size, data_spec.crop_size))
    return img.astype(np.float32)
