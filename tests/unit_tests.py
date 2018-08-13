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
            filters=4,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=False)
        input_shape = (None, 128, 128, 2)
        true = (None, 64, 64, 8)
        calc = layer.compute_output_shape(input_shape)
        assert true == calc

    def test_outputs_transpose(self):
        """test_outputs_transpose"""
        layer = conn.ComplexConv2D(
            filters=2,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True)
        input_shape = (None, 64, 64, 4)
        true = (None, 128, 128, 4)
        calc = layer.compute_output_shape(input_shape)
        assert true == calc

    def test_conv2Dforward(self):
        """test_conv2Dforward"""
        inputs = Input(shape=(128, 128, 2))
        outputs = conn.ComplexConv2D(
            filters=4,
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=False)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 64, 64, 8)
        calc = model.output_shape
        assert true == calc

    def test_conv2Dtranspose(self):
        """test_conv2Dtranspose"""
        inputs = Input(shape=(64, 64, 20))  # = 10 CDN filters
        outputs = conn.ComplexConv2D(
            filters=2,  # = 4 Keras filters
            kernel_size=3,
            strides=2,
            padding="same",
            transposed=True)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        true = (None, 128, 128, 4)
        calc = model.output_shape
        assert true == calc

    # def test_later():
        # tf.reset_default_graph()
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # tf_answer = np.array(sess.run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
