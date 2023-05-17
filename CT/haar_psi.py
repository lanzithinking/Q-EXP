#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:26:20 2023

@author: apple
"""
import numpy

from scipy import signal
'''
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    is_tensorflow_available = True
except ImportError:
    is_tensorflow_available = False
'''
is_tensorflow_available = False
if is_tensorflow_available:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    
def haar_psi_numpy(reference_image, distorted_image, preprocess_with_subsampling = True):
    """
    Calculates the HaarPSI perceptual similarity index between the two specified images. This implementation uses NumPy.
    Parameters:
    -----------
        reference_image: numpy.ndarray
            The reference image, which can be in RGB or grayscale. The values must be in the range [0, 255].
            The image must be a NumPy array of the shape (width, height, 3) in the case of RGB or a NumPy
            array in the shape (width, height) for grayscale.
        distorted_image: numpy.ndarray
            The distorted image, which is to be compared to the reference image. The image can be in RGB or
            grayscale. The values must be in the range [0, 255]. The image must be a NumPy array of the
            shape (width, height, 3) in the case of RGB or a NumPy array in the shape (width, height) for
            grayscale.
        preprocess_with_subsampling: boolean
            An optional parameter, which determines whether a preprocessing step is to be performed, which
            accommodates for the viewing distance in psychophysical experiments.
    Returns:
    --------
        (float, numpy.ndarray, numpy.ndarray): Returns a three-tuple containing the similarity score, the
        similarity maps and the weight maps. The similarity score is the Haar wavelet-based perceptual
        similarity index, measured in the interval [0,1]. The similarity maps are maps of horizontal and
        vertical local similarities. For RGB images, this variable also includes a similarity map with
        respect to the two color channels in the YIQ space. The weight maps are maps that measure the
        importance of the local similarities in the similarity maps.
    """

    # Checks if the image is a grayscale or an RGB image
    if reference_image.shape != distorted_image.shape:
        raise ValueError("The shapes of the reference image and the distorted image do not match.")
    if len(reference_image.shape) == 2:
        is_color_image = False
    elif reference_image.shape[2] == 1:
        is_color_image = False
    else:
        is_color_image = True

    # Converts the image values to double precision floating point numbers
    reference_image = reference_image.astype(numpy.float64)
    distorted_image = distorted_image.astype(numpy.float64)

    # The HaarPSI algorithm requires two constants, C and alpha, that have been experimentally determined
    # to be C = 30 and alpha = 4.2
    C = 30.0
    alpha = 4.2

    # If the images are in RGB, then they are transformed to the YIQ color space
    if is_color_image:
        reference_image_y = 0.299 * reference_image[:, :, 0] + 0.587 * reference_image[:, :, 1] + 0.114 * reference_image[:, :, 2]
        distorted_image_y = 0.299 * distorted_image[:, :, 0] + 0.587 * distorted_image[:, :, 1] + 0.114 * distorted_image[:, :, 2]
        reference_image_i = 0.596 * reference_image[:, :, 0] - 0.274 * reference_image[:, :, 1] - 0.322 * reference_image[:, :, 2]
        distorted_image_i = 0.596 * distorted_image[:, :, 0] - 0.274 * distorted_image[:, :, 1] - 0.322 * distorted_image[:, :, 2]
        reference_image_q = 0.211 * reference_image[:, :, 0] - 0.523 * reference_image[:, :, 1] + 0.312 * reference_image[:, :, 2]
        distorted_image_q = 0.211 * distorted_image[:, :, 0] - 0.523 * distorted_image[:, :, 1] + 0.312 * distorted_image[:, :, 2]
    else:
        reference_image_y = reference_image
        distorted_image_y = distorted_image

    # Subsamples the images, which simulates the typical distance between an image and its viewer
    if preprocess_with_subsampling:
        reference_image_y = subsample(reference_image_y)
        distorted_image_y = subsample(distorted_image_y)
        if is_color_image:
            reference_image_i = subsample(reference_image_i)
            distorted_image_i = subsample(distorted_image_i)
            reference_image_q = subsample(reference_image_q)
            distorted_image_q = subsample(distorted_image_q)

    # Performs the Haar wavelet decomposition
    number_of_scales = 3
    coefficients_reference_image_y = haar_wavelet_decompose(reference_image_y, number_of_scales)
    coefficients_distorted_image_y = haar_wavelet_decompose(distorted_image_y, number_of_scales)
    if is_color_image:
        coefficients_reference_image_i = numpy.abs(convolve2d(reference_image_i, numpy.ones((2, 2)) / 4.0, mode = "same"))
        coefficients_distorted_image_i = numpy.abs(convolve2d(distorted_image_i, numpy.ones((2, 2)) / 4.0, mode = "same"))
        coefficients_reference_image_q = numpy.abs(convolve2d(reference_image_q, numpy.ones((2, 2)) / 4.0, mode = "same"))
        coefficients_distorted_image_q = numpy.abs(convolve2d(distorted_image_q, numpy.ones((2, 2)) / 4.0, mode = "same"))

    # Pre-allocates the variables for the local similarities and the weights
    if is_color_image:
        local_similarities = numpy.zeros(sum([reference_image_y.shape, (3, )], ()))
        weights = numpy.zeros(sum([reference_image_y.shape, (3, )], ()))
    else:
        local_similarities = numpy.zeros(sum([reference_image_y.shape, (2, )], ()))
        weights = numpy.zeros(sum([reference_image_y.shape, (2, )], ()))

    # Computes the weights and similarities for each orientation
    for orientation in range(2):
        weights[:, :, orientation] = numpy.maximum(
            numpy.abs(coefficients_reference_image_y[:, :, 2 + orientation * number_of_scales]),
            numpy.abs(coefficients_distorted_image_y[:, :, 2 + orientation * number_of_scales])
        )
        coefficients_reference_image_y_magnitude = numpy.abs(coefficients_reference_image_y[:, :, (orientation * number_of_scales, 1 + orientation * number_of_scales)])
        coefficients_distorted_image_y_magnitude = numpy.abs(coefficients_distorted_image_y[:, :, (orientation * number_of_scales, 1 + orientation * number_of_scales)])
        local_similarities[:, :, orientation] = numpy.sum(
            (2 * coefficients_reference_image_y_magnitude * coefficients_distorted_image_y_magnitude + C) / (coefficients_reference_image_y_magnitude**2 + coefficients_distorted_image_y_magnitude**2 + C),
            axis = 2
        ) / 2

    # Computes the similarities for color channels
    if is_color_image:
        similarity_i = (2 * coefficients_reference_image_i * coefficients_distorted_image_i + C) / (coefficients_reference_image_i**2 + coefficients_distorted_image_i**2 + C)
        similarity_q = (2 * coefficients_reference_image_q * coefficients_distorted_image_q + C) / (coefficients_reference_image_q**2 + coefficients_distorted_image_q**2 + C)
        local_similarities[:, :, 2] = (similarity_i + similarity_q) / 2
        weights[:, :, 2] = (weights[:, :, 0] + weights[:, :, 1]) / 2

    # Calculates the final score
    similarity = logit(numpy.sum(sigmoid(local_similarities[:], alpha) * weights[:]) / numpy.sum(weights[:]), alpha)**2

    # Returns the result
    return similarity, local_similarities, weights


def subsample(image):
    """
    Convolves the specified image with a 2x2 mean filter and performs a dyadic subsampling step. This
    simulates the typical distance between an image and its viewer.
    Parameters:
    -----------
        image: numpy.ndarray | tensorflow.Tensor | tensorflow.Variable
            The image that is to be subsampled.
    Returns:
    --------
        numpy.ndarray | tensorflow.Tensor: Returns the subsampled image.
    """

    if is_numpy(image):
        subsampled_image = convolve2d(image, numpy.ones((2, 2)) / 4.0, mode = "same")
    elif is_tensorflow(image):
        if not is_tensorflow_available:
            raise ValueError("TensorFlow is not installed. If you have TensorFlow installed, please check your installation.")
        subsampled_image = convolve2d(image, tf.ones((2, 2)) / 4.0, mode = "same")
    else:
        raise ValueError("The image is neither a NumPy array, nor a TensorFlow tensor or variable. There are only NumPy and TensorFlow implementations available.")

    subsampled_image = subsampled_image[::2, ::2]
    return subsampled_image


def convolve2d(data, kernel, mode = "same"):
    """
    Convolves the first input array with the second one in the same way MATLAB does. Due to an
    implementation detail, the SciPy and MATLAB implementations yield different results. This method
    rectifies this shortcoming of the SciPy implementation.
    Parameters:
    -----------
        data: numpy.ndarray | tensorflow.Tensor | tensorflow.Variable
            The first input array.
        kernel: numpy.ndarray | tensorflow.Tensor | tensorflow.Variable
            The second input array with which the fist input array is being convolved.
        mode: str
            A string indicating the size of the output.
    Returns:
    --------
        numpy.ndarray | tensorflow.Tensor: Returns a 2-dimensional array containing a subset of the discrete
        linear convolution of the first input array with the second input array.
    """

    # Checks if the NumPy or the TensorFlow implementation is to be used
    if is_numpy(data) and is_numpy(kernel):

        # Due to an implementation detail of MATLAB, the input arrays have to be rotated by 90 degrees to
        # retrieve a similar result as compared to MATLAB
        rotated_data = numpy.rot90(data, 2)
        rotated_kernel = numpy.rot90(kernel, 2)

        # The convolution result has to be rotated again by 90 degrees to get the same result as in MATLAB
        result = signal.convolve2d(
            rotated_data,
            rotated_kernel,
            mode = mode
        )
        result = numpy.rot90(result, 2)

    elif is_tensorflow(data) and is_tensorflow(kernel):

        if not is_tensorflow_available:
            raise ValueError("TensorFlow is not installed. If you have TensorFlow installed, please check your installation.")

        # TensorFlow requires a 4D Tensor for convolution, the data has to be shaped [batch_size, width, height, number_of_channels]
        # and the kernel has to be shaped [width, height, number_of_channels_in, number_of_channels_out]
        data_shape = data.get_shape().as_list()
        data = tf.reshape(data, [1, data_shape[0], data_shape[1], 1])
        kernel_shape = kernel.get_shape().as_list()
        kernel = tf.reshape(kernel, [kernel_shape[0], kernel_shape[1], 1, 1])

        # Calculates the convolution, for some reason that I do not fully understand, the result has to be negated
        result = tf.nn.conv2d(
            data,
            kernel,
            padding = mode.upper(),
            strides = [1, 1, 1, 1]
        )
        result = tf.negative(tf.squeeze(result))

    else:
        raise ValueError("Either the data or the kernel is neither a NumPy array, nor a TensorFlow tensor or variable. There are only NumPy and TensorFlow implementations available.")

    # Returns the result of the convolution
    return result

def haar_wavelet_decompose(image, number_of_scales):
    """
    Performs the Haar wavelet decomposition.
    Parameters:
    -----------
        image: numpy.ndarray | tensorflow.Tensor | tensorflow.Variable
            The image that is to be decomposed.
        number_of_scales: int
            The number different filter scales that is to be used.
    Returns:
    --------
        numpy.ndarray | tensorflow.Tensor: Returns the coefficients that were determined by the Haar wavelet
        decomposition.
    """

    if is_numpy(image):

        coefficients = numpy.zeros(sum([image.shape, (2 * number_of_scales, )], ()))
        for scale in range(1, number_of_scales + 1):
            haar_filter = 2**(-scale) * numpy.ones((2**scale, 2**scale))
            haar_filter[:haar_filter.shape[0] // 2, :] = -haar_filter[:haar_filter.shape[0] // 2, :]
            coefficients[:, :, scale - 1] = convolve2d(image, haar_filter, mode = "same")
            coefficients[:, :, scale + number_of_scales - 1] = convolve2d(image, numpy.transpose(haar_filter), mode = "same")

    elif is_tensorflow(image):

        if not is_tensorflow_available:
            raise ValueError("TensorFlow is not installed. If you have TensorFlow installed, please check your installation.")

        coefficients = [None] * (2 * number_of_scales)
        for scale in range(1, number_of_scales + 1):
            upper_part = -2**(-scale) * tf.ones((2**scale // 2, 2**scale))
            lower_part = 2**(-scale) * tf.ones((2**scale // 2, 2**scale))
            haar_filter = tf.concat([upper_part, lower_part], axis = 0)
            coefficients[scale - 1] = convolve2d(image, haar_filter, mode = "same")
            coefficients[scale + number_of_scales - 1] = convolve2d(image, tf.transpose(haar_filter), mode = "same")
        coefficients = tf.stack(coefficients, axis = -1)

    else:
        raise ValueError("The image is neither a NumPy array, nor a TensorFlow tensor or variable. There are only NumPy and TensorFlow implementations available.")

    return coefficients

def sigmoid(value, alpha):
    """
    Applies the sigmoid (logistic) function to the specified value.
    Parameters:
    -----------
        value: int | float | numpy.ndarray | tensorflow.Tensor | tensorflow.Variable
            The value to which the sigmoid function is to be applied.
        alpha: float
            The steepness of the "S"-shaped curve produced by the sigmoid function.
    Returns:
    --------
        int | float | numpy.ndarray | tensorflow.Tensor: Returns the result of the sigmoid function.
    """

    if is_numpy(value):
        return 1.0 / (1.0 + numpy.exp(-alpha * value))
    elif is_tensorflow(value):
        if not is_tensorflow_available:
            raise ValueError("TensorFlow is not installed. If you have TensorFlow installed, please check your installation.")
        return 1.0 / (1.0 + tf.exp(-alpha * value))
    else:
        raise ValueError("The value is neither a NumPy array, nor a TensorFlow tensor or variable. There are only NumPy and TensorFlow implementations available.")

def logit(value, alpha):
    """
    Applies the logit function to the specified value, which is the reverse of the sigmoid
    (logistic) function.
    Parameters:
    -----------
        value: int | float | numpy.ndarray | tensorflow.Tensor | tensorflow.Variable
            The value to which the logit function is to be applied.
        alpha: float
            The steepness of the "S"-shaped curve produced by the logit function.
    Returns:
    --------
        int | float | tensorflow.Tensor: Returns the result of the logit function.
    """

    if is_numpy(value):
        return numpy.log(value / (1 - value)) / alpha
    elif is_tensorflow(value):
        if not is_tensorflow_available:
            raise ValueError("TensorFlow is not installed. If you have TensorFlow installed, please check your installation.")
        return tf.log(value / (1 - value)) / alpha
    else:
        raise ValueError("The value is neither a NumPy array, nor a TensorFlow tensor or variable. There are only NumPy and TensorFlow implementations available.")

def is_numpy(value):
    """
    Determines whether the specified value is a NumPy value, i.e. an numpy.ndarray or a NumPy scalar, etc.
    Parameters:
    -----------
        value:
            The value for which is to be determined if it is a NumPy value or not.
    Returns:
    --------
        boolean: Returns True if the value is a NumPy value and False otherwise.
    """

    return type(value).__module__.split(".")[0] == "numpy"

def is_tensorflow(value):
    """
    Determines whether the specified value is a TensorFlow value, i.e. an tensorflow.Variable or a
    tensorflow.Tensor, etc.
    Parameters:
    -----------
        value:
            The value for which is to be determined if it is a TensorFlow value or not.
    Returns:
    --------
        boolean: Returns True if the value is a TensorFlow value and False otherwise.
    """

    if not is_tensorflow_available:
        raise ValueError("TensorFlow is not installed. If you have TensorFlow installed, please check your installation.")

    return type(value).__module__.split(".")[0] == "tensorflow"