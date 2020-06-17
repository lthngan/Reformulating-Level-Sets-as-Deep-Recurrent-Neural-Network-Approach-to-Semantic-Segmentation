# --------------------------------------------------------
# Level-set RNN segmentation
# Level-set layer
# Copyright (c) 2016, Khoa Luu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import cv2
import numpy as np
import yaml
import sys
#from mnc_config import cfg


class LevelSetLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)

        self._input_img_dim = bottom[0].data.shape[1]
        self._input_state_dim = bottom[1].data.shape[1]
        # if cfg.TEST.HAS_RPN:
        #     self._input_state_dim = bottom[1].data.shape[1]
        # else:
        #     if len(bottom[1].data.shape) > 2:
        #         self._input_state_dim = bottom[1].data.shape[2]
        #     else:
        #         self._input_state_dim = bottom[1].data.shape[1]

        self.eps = float(layer_params['eps'])
        self.image_shape = (np.sqrt(self._input_img_dim), np.sqrt(self._input_img_dim))

        # add weights and biases to blobs data
        if len(self.blobs) > 0:
            print "Skipping parameter initialization"
        else:
            for i in range(0, 4):
                self.blobs.add_blob()

        # """
        ## initialize weights and biases
        # weight U
        self.blobs[0].reshape(self._input_state_dim, self._input_state_dim)
        self.blobs[0].data[...] = np.random.uniform(-np.sqrt(1. / self._input_state_dim), np.sqrt(1. / self._input_state_dim),
                                         (self._input_state_dim, self._input_state_dim))
        # weight W
        self.blobs[1].reshape(self._input_state_dim, self._input_state_dim)
        self.blobs[1].data[...] = np.random.uniform(-np.sqrt(1. / self._input_state_dim), np.sqrt(1. / self._input_state_dim),
                                         (self._input_state_dim, self._input_state_dim))
        # bias b
        self.blobs[2].reshape(self._input_state_dim)
        #self.blobs[2].data[...] = np.zeros(self._num_units)

        # bias c
        self.blobs[3].reshape(self._input_state_dim)
        #self.blobs[3].data[...] = np.zeros(self._num_units)
        # """

        print "Levelset setup call !!!"

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        self.batch_size = bottom[0].data.shape[0]

        # init output shapes
        top[0].reshape(self.batch_size, self._input_img_dim)  # output x_t
        top[1].reshape(self.batch_size, self._input_state_dim)  # output phi

        # if cfg.TEST.HAS_RPN:
        #     top[1].reshape(self.batch_size, self._input_state_dim)  # output phi
        # else:
        #     if len(bottom[1].data.shape) > 2:
        #         top[1].reshape(bottom[1].data.shape[1], self._input_state_dim)  # output phi
        #     else:
        #         top[1].reshape(bottom[1].data.shape[0], self._input_state_dim)  # output phi

    def c_func(self, x_t, phi):
        in_idx = (phi >= 0).astype(np.float32)
        out_idx = (phi < 0).astype(np.float32)

        # eps = 1e-5
        temp = np.pi * phi / self.eps

        H = 1 / 2 * (1 + phi / self.eps + (1 / np.pi) * np.sin(temp))
        H_n = 1 - H

        H1 = (phi > self.eps).astype(np.float32)  # self.phi * tf.to_float(self.phi >= self.eps)
        H3_1 = ((phi < self.eps) & (phi > -self.eps)).astype(np.float32)
        H2 = 1 - H1  # self.phi * tf.to_float(self.phi <= -self.eps)
        H3 = H * H3_1
        H4 = H_n * H3_1

        # for the inside ?
        sum1 = np.sum(x_t * H1)
        sum2 = np.sum(x_t * H3)
        c1 = (sum1 + sum2) / (np.sum(
            in_idx) + sys.float_info.epsilon)  # (sum1 + sum2)/(np.sum(idx1) + np.sum(idx3) + self.eps)

        # for the outside ?
        sum1 = np.sum(x_t * H2)
        sum2 = np.sum(x_t * H4)
        c2 = (sum1 + sum2) / (np.sum(
            out_idx) + sys.float_info.epsilon)  # (sum1 + sum2)/(np.sum(idx2) + np.sum(idx3) + self.eps)

        return c1, c2

    def forward(self, bottom, top):
        # Get current data
        self.x_t = bottom[0].data

        # if bottom[1].shape[0] == 1:
        #     self.phi = np.repeat(bottom[1].data, bottom[0].data.shape[0], axis=0)
        # else:
        self.phi = bottom[1].data
        # if cfg.TEST.HAS_RPN:
        #     self.phi = bottom[1].data
        # else:
        #     if len(bottom[1].data.shape) > 2:
        #         self.phi = np.reshape(bottom[1].data, (bottom[1].data.shape[1], bottom[1].data.shape[2]))
        #     else:
        #         self.phi = bottom[1].data


        # check match size
        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            print "Shape not matched between x_t - {} and phi - {}".format(bottom[0].data.shape[0], bottom[1].data.shape[0])

        #self.c1, self.c2 = self.c_func(self.x_t, self.phi)
        #"""
        in_idx = (self.phi >= 0).astype(np.float32)
        out_idx = (self.phi < 0).astype(np.float32)

        # eps = 1e-5
        temp = np.pi * self.phi / self.eps

        H = 1 / 2 * (1 + self.phi / self.eps + (1 / np.pi) * np.sin(temp))
        # H = 1/2 * (1 + self.phi/eps)
        # H = tf.nn.sigmoid(self.phi + self.eps)
        H_n = 1 - H

        H1 = (self.phi > self.eps).astype(np.float32)  # self.phi * tf.to_float(self.phi >= self.eps)
        H3_1 = ((self.phi < self.eps) & (self.phi > -self.eps)).astype(np.float32)
        H2 = 1 - H1  # self.phi * tf.to_float(self.phi <= -self.eps)
        H3 = H * H3_1
        H4 = H_n * H3_1

        # for the inside ?
        sum1 = np.sum(self.x_t * H1)
        sum2 = np.sum(self.x_t * H3)
        self.c1 = (sum1 + sum2) / (np.sum(
            in_idx) + sys.float_info.epsilon)  # (sum1 + sum2)/(np.sum(idx1) + np.sum(idx3) + self.eps)

        # for the outside ?
        sum1 = np.sum(self.x_t * H2)
        sum2 = np.sum(self.x_t * H4)
        self.c2 = (sum1 + sum2) / (np.sum(
            out_idx) + sys.float_info.epsilon)  # (sum1 + sum2)/(np.sum(idx2) + np.sum(idx3) + self.eps)
        #"""

        # in_idx = tf.to_float(self.phi >= 0)
        # out_idx = tf.to_float(self.phi < 0)
        #
        # H = heaviside(self.phi, 1e-5)
        #
        # c1 = np.sum(np.multiply(self.x_t, H)) / (np.sum(in_idx) + self.eps)
        # c2 = np.sum(np.multiply(self.x_t, H)) / (np.sum(out_idx) + self.eps)

        self.L1 = np.square(self.x_t - np.ones_like(self._input_img_dim, dtype=np.float32) * self.c1)
        self.L2 = np.square(self.x_t - np.ones_like(self._input_img_dim, dtype=np.float32) * self.c2)

        phi1 = np.reshape(self.phi, [self.phi.shape[0], self.image_shape[0], self.image_shape[1], 1])  # convert to matrix
        # offset_2 = self.filter_shape[0]//2
        # offset_3 = self.filter_shape[1]//2

        phix = np.zeros_like(phi1)
        phiy = np.zeros_like(phi1)
        phixx = np.zeros_like(phi1)
        phiyy = np.zeros_like(phi1)
        phixy = np.zeros_like(phi1)

        for i in range(0, self.phi.shape[0]):
            phix[i, :, :] = np.gradient(phi1[i, :, :], edge_order=1, axis=0)
            # phix = tf.nn.conv2d(phi1, self.Gx, [1, 1, 1, 1], 'SAME')
            # phix = phix0[:, offset_2:offset_2 + self.image_shape[0], offset_3:offset_3+self.image_shape[1], :]
            phiy[i, :, :] = np.gradient(phi1[i, :, :], edge_order=1, axis=1)
            # phiy = tf.nn.conv2d(phi1, self.Gy, [1, 1, 1, 1], 'SAME')
            # phiy = phiy0[:, offset_2:offset_2 + self.image_shape[0], offset_3:offset_3+self.image_shape[1], :]
            phixx[i, :, :] = np.gradient(phix[i, :, :], edge_order=1, axis=0)
            # phixx = tf.nn.conv2d(phix, self.Gx, [1, 1, 1, 1], 'SAME')
            # phixx = phixx0[:, offset_2:offset_2 + self.image_shape[0], offset_3:offset_3+self.image_shape[1], :]
            phiyy[i, :, :] = np.gradient(phiy[i, :, :], edge_order=1, axis=1)
            # phiyy = tf.nn.conv2d(phiy, self.Gy, [1, 1, 1, 1], 'SAME')
            # phiyy = phiyy0[:, offset_2:offset_2 + self.image_shape[0], offset_3:offset_3+self.image_shape[1], :]
            phixy[i, :, :] = np.gradient(phix[i, :, :], edge_order=1, axis=1)
            # phixy = tf.nn.conv2d(phix, self.Gy, [1, 1, 1, 1], 'SAME')
            # phixy = phixy0[:, offset_2:offset_2 + self.image_shape[0], offset_3:offset_3+self.image_shape[1], :]

        G = np.sqrt(np.square(phix) + np.square(phiy))

        K1 = np.multiply(phixx, np.square(phiy)) - 2 * np.multiply(np.multiply(phixy, phix), phiy) + np.multiply(phiyy,
                                                                                                                 np.square(
                                                                                                                     phix))
        K2 = np.power(np.square(phix) + np.square(phiy) + self.eps, 1.5) + sys.float_info.epsilon
        K = K1 / K2
        KG0 = K * G
        Kappa = KG0 / (np.max(KG0) + sys.float_info.epsilon)
        KG = np.reshape(Kappa, [self.phi.shape[0], self._input_state_dim])
        force = KG - (np.dot(self.L1, self.blobs[0].data) + self.blobs[2].data) + (
            np.dot(self.L2, self.blobs[1].data) + self.blobs[3].data)
        force = force / (np.max(force) + sys.float_info.epsilon)       # replace with softmax
        #force = softmax(force)

        # update s_t1
        # next_phi = self.activation_fn(
        #    np.reshape(tf.nn.xw_plus_b(prev, softmax_V, softmax_d), [self.batch_size, self._input_img_dim]))
        # next_inp = force  # do_something(prev, next_pose)  # modify this!!! 

        top[0].data[...] = force

        top[1].reshape(*self.phi.shape)
        top[1].data[...] = self.phi

        #print "Computed forward LS layer!!!"
        # """
        # top[0].reshape(*bottom[0].shape)
        # top[0].data[...] = bottom[0].data
        # pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass
        #"""
        delta_x = top[0].diff
        delta_state = top[1].diff

        # gradient w.r.t weights
        # if self.param_propagate_down[0]:
        # dL/dU
        self.blobs[0].diff[:] = np.negative(delta_x.T.dot(self.L1))

        # if self.param_propagate_down[1]:
        # dL/dW
        self.blobs[1].diff[:] = delta_x.T.dot(self.L2)

        # gradient w.r.t bias
        # if param_propagate_down[2]:
        self.blobs[2].diff[:] = np.negative(np.sum(delta_x, axis=0))
        self.blobs[3].diff[:] = np.sum(delta_x, axis=0)
        #"""
        """
        # gradient w.r.t bottom data
        if propagate_down[0]:
            dL1 = -delta_x.dot(self.blobs[0].data)
            dL2 = delta_x.dot(self.blobs[1].data)
            #dc1, dc2 = self.c_func(np.ones_like(self.phi), self.phi)
            # for the input x_t     # Dim error
            bottom[0].diff[:] = top[0].diff[:]#dL1 * np.multiply(2.0, (self.c1 - bottom[0].data)) + dL2 * np.multiply(2.0, (bottom[0].data - self.c2))
            #+ dL1 * np.multiply(2.0, (self.c1 - bottom[0].data)) + dL2 * np.multiply(2.0, (self.c2 - bottom[0].data))
            # for the state
            bottom[1].diff[:] = top[1].diff[:]#delta_x
            #+ dL1 * np.multiply(2.0, (self.c1 - bottom[0].data)) * bottom[0].data + dL2 * np.multiply(2.0, (self.c2 - bottom[0].data)) * bottom[0].data
        # """

        #print "Computed backward LS layer !!!"
