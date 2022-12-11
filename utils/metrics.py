import numpy as np
import tensorflow as tf

def dice_coefficient(true_images, completed_image):
    '''
    Dice coeffcient models how similar two images are, factoring in true positives,
    false positives, and false negatives; this is to say, this metric not only
    measures how many true positives are found, but also penalizes you for the
    number of false positives/negatives found (similar to precision). 
    '''
    true_f = tf.image.resize(tf.expand_dims(true_images, axis=0), (8, 8))
    complete_f = tf.image.resize(tf.expand_dims(completed_image[0], axis=0), (8, 8))

    true_f = np.array(tf.reshape(true_f, [-1]))
    complete_f = np.array(tf.reshape(complete_f, [-1]))

    # Let 'moe' represent the margin of error in terms of distance of pixel value allowed
    moe = 25
    diff_f = true_f - complete_f
    m1 = -moe < diff_f
    m2 = diff_f < moe
    intersection = np.sum(np.where((m1), true_f, 0))
    intersection = np.sum(np.where((m2), true_f, 0))
    true_sum = np.sum(true_f)
    completed_sum = np.sum(complete_f)

    # We use epsilon in the rare event that there 
    epsilon = 0.0001

    dice_coeff = (2. * intersection + epsilon) / (np.sum(true_sum) + np.sum(completed_sum) + epsilon)
    return dice_coeff