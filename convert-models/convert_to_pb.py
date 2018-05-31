import tensorflow as tf
import re
import sys
sys.path.append("caffe-tensorflow/examples/imagenet/models")
sys.path.append("caffe-tensorflow")
from alexnet import AlexNet

weights_path = 'bvlc_alexnet.npy'
output_graph = 'bvlc_alexnet.pb'
output_node_names = 'prob'

def __process_images(images):
    # convert from RGB to BGR
    images = images[:, :, :, ::-1]
    # Rescale
    images = tf.image.resize_images(images, (227, 227))
    return tf.to_float(images)


def __load_model():
    input = tf.placeholder(tf.float32,
                                     shape=(None, None, None, 3), name='input')
    processed_images = __process_images(input)
    # Construct and return the model
    return input, AlexNet({'data': processed_images})


with tf.Session(graph=tf.Graph()) as sess:
    input, net = __load_model()
    net.load(weights_path, sess)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names.split(",")
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
