#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for deep complex networks"""
import unittest

from keras.layers import Input
from keras.models import Model
# import tensorflow as tf

# import numpy as np
# import tensorflow as tf

import complexnn as conn


class TestDNCMethods(unittest.TestCase):
    """Unit test class"""

    def test_outputs_forward(self):
        """test_outputs_forward"""
        layer = conn.ComplexConv2D(
            filters=2,
            kernel_size=3,
            strides=2,
            padding="same")
        outputs_shape = layer.compute_output_shape((None, 128, 128, 0))
        print(f'outputs shape: {outputs_shape}')

    def test_conv2Dforward(self):
        """test_conv2Dforward"""
        inputs = Input(shape=(128, 128, 2))
        outputs = conn.ComplexConv2D(
            1, 3, 1, padding="same",
            activation="relu", transposed=False)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

    # def test_conv2Dtranspose(self):
    #     """test_conv2Dtranspose"""

        # Transposed convolution
        layer = conn.ComplexConv2D(
            filters=2,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            transposed=True)
        outputs_shape = layer.compute_output_shape((None, 64, 64, 4))
        print(f'outputs shape: {outputs_shape}')
    #     inputs = Input(shape=(5, 5, 0))
    #     outputs = conn.ComplexConv2D(
    #         1, 3, 1, padding="same",
    #         activation="relu", transposed=True)(inputs)
    #     model = Model(inputs=inputs, outputs=outputs)
    #     model.summary()

    # def test_later():
        # tf.reset_default_graph()
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # tf_answer = np.array(sess.run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
