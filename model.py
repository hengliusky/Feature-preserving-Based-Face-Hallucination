import time

import tensorflow as tf
import tensorlayer as tl

from utils import Res_block, Pixelshuffler


def encoder(input_imgs, is_train=True, reuse=False):
    '''
    input_imgs: the input images to be encoded into a vector as latent representation. size here is [b_size,64,64,3]
    '''
    z_dim = 128  # 512
    ef_dim = 64  # encoder filter number

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = tl.layers.InputLayer(
            input_imgs, name='en/in')  # (b_size,64,64,3)
        net_h0 = tl.layers.Conv2d(
            net_in,
            n_filter=ef_dim,
            filter_size=5,
            strides=2,
            act=None,
            padding='SAME',
            W_init=w_init,
            name='en/h0/conv2d')
        net_h0 = tl.layers.BatchNormLayer(
            net_h0,
            act=tf.nn.relu,
            is_train=is_train,
            gamma_init=gamma_init,
            name='en/h0/batch_norm')
        # net_h0.outputs._shape = (b_size,32,32,64)

        net_h1 = tl.layers.Conv2d(
            net_h0,
            n_filter=ef_dim * 2,
            filter_size=5,
            strides=2,
            act=None,
            padding='SAME',
            W_init=w_init,
            name='en/h1/conv2d')
        net_h1 = tl.layers.BatchNormLayer(
            net_h1,
            act=tf.nn.relu,
            is_train=is_train,
            gamma_init=gamma_init,
            name='en/h1/batch_norm')
        # net_h1.outputs._shape = (b_size,16,16,64*2)

        net_h2 = tl.layers.Conv2d(
            net_h1,
            n_filter=ef_dim * 4,
            filter_size=5,
            strides=2,
            act=None,
            padding='SAME',
            W_init=w_init,
            name='en/h2/conv2d')
        net_h2 = tl.layers.BatchNormLayer(
            net_h2,
            act=tf.nn.relu,
            is_train=is_train,
            gamma_init=gamma_init,
            name='en/h2/batch_norm')
        # net_h2.outputs._shape = (b_size,8,8,64*4)

        net_h3 = tl.layers.Conv2d(
            net_h2,
            n_filter=ef_dim * 8,
            filter_size=5,
            strides=2,
            act=None,
            padding='SAME',
            W_init=w_init,
            name='en/h3/conv2d')
        net_h3 = tl.layers.BatchNormLayer(
            net_h3,
            act=tf.nn.relu,
            is_train=is_train,
            gamma_init=gamma_init,
            name='en/h3/batch_norm')
        # net_h2.outputs._shape = (b_size,4,4,64*8)

        # mean of z
        net_h4 = tl.layers.FlattenLayer(net_h3, name='en/h4/flatten')
        # net_h4.outputs._shape = (b_size,8*8*64*4)
        net_out1 = tl.layers.DenseLayer(
            net_h4,
            n_units=z_dim,
            act=tf.identity,
            W_init=w_init,
            name='en/h3/lin_sigmoid')
        net_out1 = tl.layers.BatchNormLayer(
            net_out1,
            act=tf.identity,
            is_train=is_train,
            gamma_init=gamma_init,
            name='en/out1/batch_norm')

        # net_out1 = DenseLayer(net_h4, n_units=z_dim, act=tf.nn.relu,
        #         W_init = w_init, name='en/h4/lin_sigmoid')
        z_mean = net_out1.outputs  # (b_size,512)

        # log of variance of z(covariance matrix is diagonal)
        net_h5 = tl.layers.FlattenLayer(net_h3, name='en/h5/flatten')
        net_out2 = tl.layers.DenseLayer(
            net_h5,
            n_units=z_dim,
            act=tf.identity,
            W_init=w_init,
            name='en/h4/lin_sigmoid')
        net_out2 = tl.layers.BatchNormLayer(
            net_out2,
            act=tf.nn.softplus,
            is_train=is_train,
            gamma_init=gamma_init,
            name='en/out2/batch_norm')
        # net_out2 = DenseLayer(net_h5, n_units=z_dim, act=tf.nn.relu,
        #         W_init = w_init, name='en/h5/lin_sigmoid')
        z_log_sigma_sq = net_out2.outputs + 1e-6  # (b_size,512)

        network = tl.layers.merge_networks([net_h3, net_out1, net_out2])

    return network, z_mean, z_log_sigma_sq, net_h2.outputs


def code_discriminator(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope('code_discriminator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_h0 = tl.layers.InputLayer(t_image, name='input')

        net_h0 = tl.layers.Conv2d(
            net_h0,
            n_filter=64,
            filter_size=3,
            strides=1,
            padding='SAME',
            W_init=w_init,
            name='h0/conv')  #(b_size,8,8,64*4)

        net_h1 = tl.layers.Conv2d(
            net_h0,
            n_filter=64,
            filter_size=5,
            strides=2,
            padding='SAME',
            W_init=w_init,
            name='h1/conv')
        net_h1 = tl.layers.BatchNormLayer(
            net_h1,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h1/batch_norm')

        net_h2 = tl.layers.Conv2d(
            net_h1,
            n_filter=64 * 2,
            filter_size=3,
            strides=1,
            padding='SAME',
            W_init=w_init,
            name='h2/conv1')
        net_h2 = tl.layers.BatchNormLayer(
            net_h2,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h2/batch_norm1')
        net_h2 = tl.layers.Conv2d(
            net_h2,
            n_filter=64 * 2,
            filter_size=5,
            strides=2,
            padding='SAME',
            W_init=w_init,
            name='h2/conv2')
        net_h2 = tl.layers.BatchNormLayer(
            net_h2,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h2/batch_norm2')

        net_h3 = tl.layers.Conv2d(
            net_h2,
            n_filter=64 * 4,
            filter_size=3,
            strides=1,
            padding='SAME',
            W_init=w_init,
            name='h3/conv1')
        net_h3 = tl.layers.BatchNormLayer(
            net_h3,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h3/batch_norm1')
        net_h3 = tl.layers.Conv2d(
            net_h3,
            n_filter=64 * 4,
            filter_size=5,
            strides=2,
            padding='SAME',
            W_init=w_init,
            name='h3/conv2')
        net_h3 = tl.layers.BatchNormLayer(
            net_h3,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h3/batch_norm2')

        net_h4 = tl.layers.Conv2d(
            net_h3,
            n_filter=64 * 8,
            filter_size=3,
            strides=1,
            padding='SAME',
            W_init=w_init,
            name='h4/conv1')
        net_h4 = tl.layers.BatchNormLayer(
            net_h4,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h4/batch_norm1')
        net_h4 = tl.layers.Conv2d(
            net_h4,
            n_filter=64 * 8,
            filter_size=5,
            strides=2,
            padding='SAME',
            W_init=w_init,
            name='h4/conv2')
        net_h4 = tl.layers.BatchNormLayer(
            net_h4,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h4/batch_norm2')

        net_out = tl.layers.FlattenLayer(net_h4, name='Flatten')
        net_out = tl.layers.DenseLayer(
            net_out, n_units=1024, act=lrelu, name='dense2')
        net_out = tl.layers.DenseLayer(
            net_out, n_units=1, act=None, name='output')

    return net_out, net_out.outputs


def generator(t_image, reuse=False):
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope('generator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = tl.layers.InputLayer(t_image, name='input/images')

        net_h0 = tl.layers.Conv2d(
            net_in,
            n_filter=32,
            filter_size=3,
            strides=1,
            act=lrelu,
            padding='SAME',
            name='h0/conv1')
        net_h0 = tl.layers.Conv2d(
            net_h0,
            n_filter=64,
            filter_size=3,
            strides=1,
            act=lrelu,
            padding='SAME',
            name='h0/conv2')
        net_h0 = tl.layers.Conv2d(
            net_h0,
            n_filter=64,
            filter_size=3,
            strides=1,
            act=lrelu,
            padding='SAME',
            name='h0/conv3')

        net_h1 = Res_block(net_h0, act=lrelu, name='h1/Res_block1')
        net_h1 = Res_block(net_h1, act=lrelu, name='h1/Res_block2')
        net_h1 = Res_block(net_h1, act=lrelu, name='h1/Res_block3')
        net_h1 = Res_block(net_h1, act=lrelu, name='h1/Res_block4')

        net_h2 = Pixelshuffler(net_h1, scale=2, act=lrelu, name='h2')

        net_h3 = Res_block(net_h2, act=lrelu, name='h3/Res_block1')
        net_h3 = Res_block(net_h3, act=lrelu, name='h3/Res_block2')
        net_h3 = Res_block(net_h3, act=lrelu, name='h3/Res_block3')

        net_h4 = Pixelshuffler(net_h3, scale=2, act=lrelu, name='h4')

        skip0 = Pixelshuffler(net_h0, scale=4, act=lrelu, name='skip0')
        net_h5 = tl.layers.ConcatLayer(
            [net_h4, skip0], concat_dim=-1, name='Concat1')
        net_h5 = tl.layers.Conv2d(
            net_h5,
            n_filter=64,
            filter_size=3,
            strides=1,
            act=lrelu,
            padding='SAME',
            name='h5/conv')

        net_h6 = Res_block(net_h5, act=lrelu, name='h6/Res_block1')
        net_h6 = Res_block(net_h6, act=lrelu, name='h6/Res_block2')
        net_h6 = Res_block(net_h6, act=lrelu, name='h6/Res_block3')

        net_h7 = Pixelshuffler(net_h6, scale=2, act=lrelu, name='h7')

        skip1 = Pixelshuffler(net_h0, scale=8, act=lrelu, name='skip1')
        skip2 = Pixelshuffler(net_h3, scale=4, act=lrelu, name='skip2')

        net_h8 = tl.layers.ConcatLayer(
            [net_h7, skip1, skip2], concat_dim=-1, name='Concat2')

        net_out = tl.layers.Conv2d(
            net_h8,
            n_filter=128,
            filter_size=3,
            strides=1,
            act=lrelu,
            padding='SAME',
            name='out/conv1')
        net_out = tl.layers.Conv2d(
            net_out,
            n_filter=64,
            filter_size=3,
            strides=1,
            act=lrelu,
            padding='SAME',
            name='out/conv2')
        net_out = tl.layers.Conv2d(
            net_out,
            n_filter=3,
            filter_size=3,
            strides=1,
            act=None,
            padding='SAME',
            name='out/conv3')
    return net_out


def discriminator(input_images, is_train=True, reuse=False):
    g_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    with tf.variable_scope('discriminator', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = tl.layers.InputLayer(input_images, name='input/images')
        net_h0 = tl.layers.Conv2d(
            net_in,
            n_filter=32,
            filter_size=5,
            strides=2,
            padding='SAME',
            name='h0/conv')
        net_h0 = tl.layers.BatchNormLayer(
            net_h0,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h0/batch_norm')

        net_h1 = tl.layers.Conv2d(
            net_h0,
            n_filter=64,
            filter_size=5,
            strides=2,
            padding='SAME',
            name='h1/conv')
        net_h1 = tl.layers.BatchNormLayer(
            net_h1,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h1/batch_norm')

        net_h2 = tl.layers.Conv2d(
            net_h1,
            n_filter=128,
            filter_size=5,
            strides=2,
            padding='SAME',
            name='h2/conv')
        net_h2 = tl.layers.BatchNormLayer(
            net_h2,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h2/batch_norm')

        net_h3 = tl.layers.Conv2d(
            net_h2,
            n_filter=256,
            filter_size=5,
            strides=2,
            padding='SAME',
            name='h3/conv')
        net_h3 = tl.layers.BatchNormLayer(
            net_h3,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h3/batch_norm')

        net_h4 = tl.layers.Conv2d(
            net_h3,
            n_filter=32,
            filter_size=3,
            strides=1,
            padding='SAME',
            name='h4/conv')
        net_h4 = tl.layers.BatchNormLayer(
            net_h4,
            act=lrelu,
            is_train=is_train,
            gamma_init=g_init,
            name='h4/batch_norm')

        net_ho = tl.layers.FlattenLayer(net_h4, name='ho/flatten')
        net_ho = tl.layers.DenseLayer(
            net_ho, n_units=1024, act=tf.identity, name='ho/dense_1')
        net_ho = tl.layers.DenseLayer(
            net_ho, n_units=1, act=tf.identity, name='ho/dense_2')

        logits = net_ho.outputs

    return net_ho, logits


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse):
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(
            [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ],
            axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = tl.layers.InputLayer(bgr, name='input')
        """ conv1 """
        network = tl.layers.Conv2d(
            net_in,
            n_filter=64,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv1_1')
        out1 = network.outputs
        network = tl.layers.Conv2d(
            network,
            n_filter=64,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv1_2')
        network = tl.layers.MaxPool2d(
            network, filter_size=2, strides=2, padding='SAME', name='pool1')
        """ conv2 """
        network = tl.layers.Conv2d(
            network,
            n_filter=128,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv2_1')
        out2 = network.outputs
        network = tl.layers.Conv2d(
            network,
            n_filter=128,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv2_2')
        network = tl.layers.MaxPool2d(
            network, filter_size=2, strides=2, padding='SAME', name='pool2')
        """ conv3 """
        network = tl.layers.Conv2d(
            network,
            n_filter=256,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv3_1')
        out3 = network.outputs
        network = tl.layers.Conv2d(
            network,
            n_filter=256,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv3_2')
        network = tl.layers.Conv2d(
            network,
            n_filter=256,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv3_3')
        network = tl.layers.Conv2d(
            network,
            n_filter=256,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv3_4')
        network = tl.layers.MaxPool2d(
            network, filter_size=2, strides=2, padding='SAME', name='pool3')
        """ conv4 """
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv4_1')
        out4 = network.outputs
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv4_2')
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv4_3')
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv4_4')
        network = tl.layers.MaxPool2d(
            network, filter_size=2, strides=2, padding='SAME',
            name='pool4')  # (batch_size, 14, 14, 512)
        """ conv5 """
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv5_1')
        out5 = network.outputs
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv5_2')
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv5_3')
        network = tl.layers.Conv2d(
            network,
            n_filter=512,
            filter_size=3,
            strides=1,
            act=tf.nn.relu,
            padding='SAME',
            name='conv5_4')
        network = tl.layers.MaxPool2d(
            network, filter_size=2, strides=2, padding='SAME',
            name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DenseLayer(
            network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = tl.layers.DenseLayer(
            network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = tl.layers.DenseLayer(
            network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, [out1, out2, out3, out4, out5]
