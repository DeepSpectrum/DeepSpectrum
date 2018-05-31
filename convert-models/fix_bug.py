with open('caffe-tensorflow/kaffe/caffe/resolver.bug', 'r') as bugged, open('caffe-tensorflow/kaffe/caffe/resolver.py', 'w') as fixed:
    for line in bugged:
        first_fix = line.replace('from . import caffepb', 'from . import caffe_pb2')
        second_fix = first_fix.replace('self.caffepb = caffepb', 'self.caffepb = caffe_pb2')
        fixed.write(second_fix)


with open('caffe-tensorflow/kaffe/tensorflow/network.bug', 'r') as bugged, open('caffe-tensorflow/kaffe/tensorflow/network.py', 'w') as fixed:
    for line in bugged:

        first_fix = line.replace('input_groups = tf.split(3, group, input)', 'input_groups = tf.split(input, group, 3)')
        second_fix = first_fix.replace('kernel_groups = tf.split(3, group, kernel)', 'kernel_groups = tf.split(kernel, group, 3)')
        third_fix = second_fix.replace('output = tf.concat(3, output_groups)', 'output = tf.concat(output_groups, 3)')
        fixed.write(third_fix)