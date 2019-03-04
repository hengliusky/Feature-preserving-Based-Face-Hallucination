import os
import time

import numpy as np
import scipy.misc
import skimage
import tensorflow as tf
import tensorlayer as tl

from config import *
from model import generator
from utils import *


def rgb2ycbcr(rgb_img):
    mat = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0],
                    [112.0, -93.786, -18.214]])
    offset = np.array([16, 128, 128])
    ycbcr_img = np.zeros(rgb_img.shape)
    rgb_img = (rgb_img + 1.0) / 2.0
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(
                np.dot(mat, rgb_img[x, y, :]) + offset)
    return ycbcr_img[:, :, 0]


def ymse(im1, im2):
    im1_y = rgb2ycbcr(im1)
    im2_y = rgb2ycbcr(im2)
    return np.square(im1_y - im2_y).mean()


def psnr(im1, im2):
    mse = ymse(im1, im2)
    psnr = 10 * np.log10((255. * 255.) / mse)
    return psnr


def read_image(img_path=''):
    image = scipy.misc.imread(img_path, mode='RGB')
    h, w = image.shape[:2]
    crop_h = crop_w = 148
    cx1 = cx2 = int(round((w - crop_w) / 2.))
    cy1 = cy2 = int(round((h - crop_h) / 2.))
    hr_image = tl.prepro.imresize(
        image[cx1:h - cy1, cx2:w - cy2],
        size=[128, 128],
        interp='bicubic',
        mode=None)
    hr_image = hr_image / (255. / 2.) - 1.
    lr_image = tl.prepro.imresize(
        hr_image, size=[16, 16], interp='bicubic', mode=None)
    lr_image = lr_image / (255. / 2.) - 1.
    bic_image = tl.prepro.imresize(
        lr_image, size=[128, 128], interp='bicubic', mode=None)
    bic_image = bic_image / (255. / 2.) - 1.
    return lr_image, hr_image, bic_image


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tl.files.exists_or_mkdir(save_img_dir)
    t_image = tf.placeholder(
        'float32', [1, input_size, input_size, channels], name='t_image_input')
    t_bicubic = tf.image.resize_images(
        t_image, size=[128, 128], method=2, align_corners=False)
    net_g = generator(t_image, reuse=False)
    t_est = net_g.outputs + t_bicubic
    sess = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    tl.files.load_and_assign_npz(
        sess=sess,
        name=checkpoint_dir + '/generator/generator.npz',
        network=net_g)
    total_ymse, total_psnr, total_ssim = 0, 0, 0
    for idx in range(18001, 18101):
        start_time = time.time()
        file_name = str(idx).zfill(6) + ".jpg"
        file_path = os.path.join(test_image_path, file_name)
        lr_image, hr_image, bic_image = read_image(file_path)
        # print(file_name)

        sr_image = sess.run(t_est, {t_image: [lr_image]})
        # print('time=%.5f' % (time.time() - start_time))
        # squeeze the firt axis of hr_image, bic_image, sr_image
        sr_face = np.squeeze(sr_image, axis=0)
        hr_face = np.squeeze(hr_image, axis=0)
        process_time = time.time() - start_time

        # compute MSE and PSNR
        # ymse, psnr = compute_psnr(sr_face, hr_face)
        y_sr_face = rgb2ycbcr(sr_face)
        y_hr_face = rgb2ycbcr(hr_face)
        ymse = skimage.measure.compare_mse(y_hr_face, y_sr_face)
        psnr = skimage.measure.compare_psnr(
            y_hr_face, y_sr_face, data_range=255)
        ssim = skimage.measure.compare_ssim(
            y_hr_face, y_sr_face, data_range=255)

        print("Image: %d, ymse: %.4f, psnr: %.4f, ssim: %.4f, time: %.4f" % (
            (idx + 18001), ymse, psnr, ssim, process_time))
        total_psnr += psnr
        total_ymse += ymse
        total_ssim += ssim

        # save the generated face and hr face
        # stack_image = np.column_stack([sr_face, hr_face])
        # tl.vis.save_image(stack_image,
        #                   args.outf + str(idx + 18001).zfill(6) + '.png')

    print("average ymse: %.4f, average psnr: %.4f, average ssim: %.4f" %
          (total_ymse / 2000, total_psnr / 2000, total_ssim / 2000))


if __name__ == "__main__":
    main()
