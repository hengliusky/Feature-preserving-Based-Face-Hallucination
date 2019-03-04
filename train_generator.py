import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from config import *
from model import generator, Vgg19_simple_api
from utils import *


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx:idx + n_threads]
        b_imgs = tl.prepro.threading_data(
            b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    tl.files.exists_or_mkdir(checkpoint_dir + '/generator')
    tl.files.exists_or_mkdir(summary_dir + '/generator')

    img_list = sorted(
        tl.files.load_file_list(
            path=train_image_path, regx='.*.jpg', printable=False))

    train_hr_imgs = read_all_imgs(
        img_list, path=train_image_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder(
        'float32', [batch_size, input_size, input_size, channels],
        name='t_image_input')
    t_target_image = tf.placeholder(
        'float32', [batch_size, label_size, label_size, channels],
        name='t_target_image')

    t_bicubic = tf.image.resize_images(
        t_image, size=[128, 128], method=2, align_corners=False)
    net_g = generator(t_image, reuse=False)
    t_est = net_g.outputs + t_bicubic

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224 = tf.image.resize_images(
        t_est, size=[224, 224], method=0, align_corners=False)
    net_vgg, vgg_target_emb = Vgg19_simple_api(
        (t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api(
        (t_predict_image_224 + 1) / 2, reuse=True)

    ###=============================MSE LOSS=============================###
    mse_loss = tf.reduce_mean(tf.square(t_est - t_target_image))
    ###==================================================================###

    ###=============================VGG LOSS=============================###
    vgg1_loss = 1e-4 * tf.reduce_mean(
        tf.square(vgg_target_emb[0] - vgg_predict_emb[0]))
    vgg2_loss = 1e-6 * tf.reduce_mean(
        tf.square(vgg_target_emb[2] - vgg_predict_emb[2]))
    vgg3_loss = 2e-6 * tf.reduce_mean(
        tf.square(vgg_target_emb[4] - vgg_predict_emb[4]))
    vgg_loss = vgg1_loss + vgg2_loss + vgg3_loss
    ###==================================================================###

    with tf.name_scope('summary'):
        tf.summary.image('bicubic', t_bicubic, max_outputs=3)
        tf.summary.image('generator', net_g.outputs, max_outputs=3)
        tf.summary.image('t_est', t_est, max_outputs=3)
        tf.summary.image('targets', t_target_image, max_outputs=3)
        tf.summary.scalar('mse_loss', mse_loss)
        tf.summary.scalar('vgg_loss', vgg_loss)
        tf.summary.scalar('total_loss', mse_loss + vgg_loss)

    g_vars = tl.layers.get_variables_with_name('generator', True, True)

    g_optim = tf.train.AdamOptimizer(
        2e-4, beta1=0.5).minimize(
            mse_loss + vgg_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    loss_writer = tf.summary.FileWriter(summary_dir + '/generator', sess.graph)
    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(
        sess=sess,
        name=checkpoint_dir + '/generator/generator.npz',
        network=net_g)

    merged = tf.summary.merge_all()
    step = 0

    for epoch in range(num_epoch):
        for idx in range(0, len(train_hr_imgs), batch_size):
            start_time = time.time()
            hr_image = tl.prepro.threading_data(
                train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn)
            lr_image = tl.prepro.threading_data(hr_image, fn=downsample_fn)
            mse, summary, _ = sess.run([mse_loss, merged, g_optim], {
                t_image: lr_image,
                t_target_image: hr_image
            })
            loss_writer.add_summary(summary, step)
            print("Epoch [%2d/%2d] %4d, mse_loss: %.8f, time: %.4f" %
                  (epoch + 1, num_epoch, step, mse, time.time() - start_time))
            step = step + 1

        tl.files.save_npz(
            net_g.all_params,
            name=checkpoint_dir + '/generator/generator.npz',
            sess=sess)
        tl.files.save_npz(
            net_g.all_params,
            name=checkpoint_dir + '/generator/generator%d.npz' % (epoch + 1),
            sess=sess)


if __name__ == '__main__':
    main()
