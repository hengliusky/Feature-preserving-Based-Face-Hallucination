import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from config import *
from model import Vgg19_simple_api, discriminator, encoder, generator
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tl.files.exists_or_mkdir(checkpoint_dir+'/gan_encoder')
    tl.files.exists_or_mkdir(summary_dir+'/gan_encoder')

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

    ## encoder inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_64 = tf.image.resize_images(
        t_target_image, size=[64, 64], method=0, align_corners=False)
    t_predict_image_64 = tf.image.resize_images(
        t_est, size=[64, 64], method=0, align_corners=False)
    net_e, _, _, enc_target_emb = encoder(
        t_target_image_64, reuse=False)
    _, _, _, enc_predict_emb = encoder(
        t_predict_image_64, reuse=True)

    net_d, logits_real = discriminator(
        t_target_image, is_train=True, reuse=False)
    _, logits_fake = discriminator(t_est, is_train=True, reuse=True)

    ###=============================GAN LOSS=============================###
    d_loss = -1e-3 * tf.reduce_mean(
        tf.log(tf.nn.sigmoid(logits_real - logits_fake) + 1e-9))
    g_gan_loss = -1e-3 * tf.reduce_mean(
        tf.log(tf.nn.sigmoid(logits_fake - logits_real) + 1e-9))
    ###==================================================================###

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

    ###===========================ENCODER LOSS===========================###
    enc_loss = tf.reduce_mean(tf.square(enc_target_emb - enc_predict_emb))
    ###==================================================================###

    g_loss = mse_loss + g_gan_loss + vgg_loss + enc_loss

    with tf.name_scope('summary'):
        tf.summary.image('generator', net_g.outputs, max_outputs=3)
        tf.summary.image('t_est', t_est, max_outputs=3)
        tf.summary.image('targets', t_target_image, max_outputs=3)
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('enc_loss', enc_loss)
        tf.summary.scalar('mse_loss', mse_loss)
        tf.summary.scalar('vgg_loss', vgg_loss)
        tf.summary.scalar('g_gan_loss', g_gan_loss)

    # e_vars = tl.layers.get_variables_with_name('encoder', True, True)
    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

    g_optim = tf.train.AdamOptimizer(
        2e-4, beta1=0.5).minimize(
            g_loss, var_list=[g_vars])
    d_optim = tf.train.AdamOptimizer(
        2e-4, beta1=0.5).minimize(
            d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    loss_writer = tf.summary.FileWriter(summary_dir + '/gan_encoder',
                                        sess.graph)
    sess.run(tf.global_variables_initializer())
    if tl.files.load_and_assign_npz(
            sess=sess, name=checkpoint_dir + '/gan_encoder/generator.npz',
            network=net_g) is False:
        tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir + '/generator/generator.npz',
            network=net_g)
    tl.files.load_and_assign_npz(
        sess=sess,
        name=checkpoint_dir + '/gan_encoder/discriminator.npz',
        network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print(
            "Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg"
        )
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    ###============================= LOAD ENCODER ===============================###
    tl.files.load_and_assign_npz(
        sess=sess, name='./VAE/checkpoint/vae_0808/net_e.npz', network=net_e)

    merged = tf.summary.merge_all()
    step = 0

    for epoch in range(num_epoch):
        for idx in range(0, len(train_hr_imgs), batch_size):
            hr_image = tl.prepro.threading_data(
                train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn)
            lr_image = tl.prepro.threading_data(hr_image, fn=downsample_fn)
            ## update D
            errD, summary, _ = sess.run([d_loss, merged, d_optim], {
                t_image: lr_image,
                t_target_image: hr_image
            })
            loss_writer.add_summary(summary, step)
            ## update G
            errG, mse, vgg, enc, g_gan, _ = sess.run(
                [g_loss, mse_loss, vgg_loss, enc_loss, g_gan_loss, g_optim], {
                    t_image: lr_image,
                    t_target_image: hr_image
                })
            print(
                "Epoch [%2d/%2d] %4d, d_loss: %.8f g_loss: %.8f(mse=%.8f, vgg=%.8f, enc=%.8f, g_gan=%.8f)"
                % (epoch + 1, num_epoch, step, errD, errG, mse, vgg, enc,
                   g_gan))
            step = step + 1

        if (epoch + 1) % 5 == 0 :
            tl.files.save_npz(
                net_g.all_params,
                name=checkpoint_dir + '/gan_encoder/generator.npz',
                sess=sess)
            tl.files.save_npz(
                net_d.all_params,
                name=checkpoint_dir + '/gan_encoder/discriminator.npz',
                sess=sess)
            tl.files.save_npz(
                net_g.all_params,
                name=checkpoint_dir + '/gan_encoder/generator%d.npz' % (epoch + 1),
                sess=sess)
            tl.files.save_npz(
                net_d.all_params,
                name=checkpoint_dir + '/gan_encoder/discriminator%d.npz' % (epoch + 1),
                sess=sess)


if __name__ == '__main__':
    main()
