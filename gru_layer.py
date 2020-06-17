# --------------------------------------------------------
# Level-set RNN segmentation
# GRU layer
# Copyright (c) 2016, Khoa Luu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import cv2
import numpy as np
import yaml

import caffe


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def tanh(x):
    return 2. * sigmoid(2. * x) - 1.


class GRULayer(caffe.Layer):
    def setup(self, bottom, top):  # bottom: input, top: output
        # assert len(bottom) == 1
        # assert bottom[0].data.ndim >= 3
        # assert len(top) == 1

        layer_params = yaml.load(self.param_str_)

        self._input_dim = bottom[0].data.shape[1]
        self._num_units = layer_params['num_units']

        # add weights and biases to blobs data
        if len(self.blobs) > 0:
            print "Skipping parameter initialization"
        else:
            for i in range(0, 3):
                self.blobs.add_blob()

        ## initialize weights and biases
        # weight Us
        self.blobs[0].reshape(3, self._input_dim, self._num_units)
        self.blobs[0].data[...] = np.random.uniform(-np.sqrt(1. / self._input_dim), np.sqrt(1. / self._input_dim),
                                          (3, self._input_dim, self._num_units))
        # weight Ws
        self.blobs[1].reshape(3, self._num_units, self._num_units)
        self.blobs[1].data[...] = np.random.uniform(-np.sqrt(1. / self._input_dim), np.sqrt(1. / self._input_dim),
                                          (3, self._input_dim, self._num_units))
        # bias bs
        self.blobs[2].reshape(3, self._num_units)
        self.blobs[2].data[...] = np.zeros((3, self._num_units))

        # self.W_z = np.zeros((self._num_units, self._num_units))
        # self.W_r = np.zeros((self._num_units, self._num_units))
        # self.W_h = np.zeros((self._num_units, self._num_units))
        #
        # self.U_z = np.zeros((self._input_dim, self._num_units))
        # self.U_r = np.zeros((self._input_dim, self._num_units))
        # self.U_h = np.zeros((self._input_dim, self._num_units))
        #
        # self.b_z = np.zeros((self._num_units,))
        # self.b_r = np.zeros((self._num_units,))
        # self.b_h = np.zeros((self._num_units,))

        # self.param_propagate_down.resize(len(self.blobs), true)         # not be working!!! We may not need this (need to add in prototxt file NOT here)

        print "GRU setup call"

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        self._batch_no = bottom[0].data.shape[0]

        # init output shapes
        top[0].reshape(self._batch_no, self._num_units)  # output x_t
        # pass

    def forward(self, bottom, top):
        """To be implemented."""
        ## Get input x and state
        self.x = bottom[0].data[...]
        self.state = bottom[1].data[...]

        # x_dim = self._num_units * 2

        # compute r and uand apply sigmoid on r and u
        self.z = sigmoid(np.dot(self.x, self.blobs[0].data[0, :, :]) + np.dot(self.state, self.blobs[1].data[0, :, :]) +
                         self.blobs[2].data[0])
        self.r = sigmoid(np.dot(self.x, self.blobs[0].data[1, :, :]) + np.dot(self.state, self.blobs[1].data[1, :, :]) +
                         self.blobs[2].data[1])

        # compute c
        self.h = tanh(np.dot(self.x, self.blobs[0].data[2, :, :]) + np.dot(np.multiply(self.r, self.state),
                                                                           self.blobs[1].data[2, :, :]) +
                      self.blobs[2].data[2])

        # compute new_state
        new_state = np.multiply(self.z, self.state) + np.multiply((1 - self.z), self.h)

        top[0].data[...] = new_state  # copy all data along all dim, ... ~ :

        #print "Computed forward GRU layer !!!"

    def backward(self, top, propagate_down,
                 bottom):  # propagate_down here is the parameters of layers (e.g weights or biases)
        """To be implemented"""
        #pass

        #"""
        # dLdU_r = np.zeros(self.U_r.shape)
        # dLdW_r = np.zeros(self.W_r.shape)
        # dLdb_r = np.zeros(self.b_r.shape)
        #
        # dLdU_z = np.zeros(self.U_z.shape)
        # dLdW_z = np.zeros(self.W_z.shape)
        # dLdb_z = np.zeros(self.b_z.shape)
        #
        # dLdU_h = np.zeros(self.U_h.shape)
        # dLdW_h = np.zeros(self.W_h.shape)
        # dLdb_h = np.zeros(self.b_h.shape)

        delta = top[0].diff[...]
        dLdh = delta * (1 - self.z) * self.h * (1 - self.h)
        dLdr = dLdh.dot(self.blobs[1].data[2]) * self.state
        dLdz = delta * (self.state - self.h) * self.z * (1 - self.z) # s[t - 2]

        # gradient w.r.t weights
        #if param_propagate_down[0]:
        # dL/dU
        self.blobs[0].diff[0] = dLdz.T.dot(self.x)
        self.blobs[0].diff[1] = dLdr.T.dot(self.x)
        self.blobs[0].diff[2] = dLdh.T.dot(self.x)

        #if param_propagate_down[1]:
        # dL/dW
        self.blobs[1].diff[0] = np.dot(dLdz.T, self.state)
        self.blobs[1].diff[1] = np.dot(dLdr.T, self.state)
        self.blobs[1].diff[2] = np.dot(dLdh.T, self.r * self.state)

        # gradient w.r.t bias
        #if param_propagate_down[2]:
        self.blobs[2].diff[0] = np.sum(dLdz, axis=0)
        self.blobs[2].diff[1] = np.sum(dLdr, axis=0)
        self.blobs[2].diff[2] = np.sum(dLdh, axis=0)

        # gradient w.r.t bottom data
        if propagate_down[0]:
            bottom[0].diff[:] = dLdh.dot(self.blobs[1].data[2]) + dLdr.dot(self.blobs[1].data[1]) + dLdz.dot(self.blobs[0].data[0])
            bottom[1].diff[:] = dLdh.dot(self.blobs[1].data[2]) * self.r + dLdr.dot(self.blobs[1].data[1]) + dLdz.dot(self.blobs[0].data[0])
        #"""
        #print "Computed backward GRU layer !!!"

