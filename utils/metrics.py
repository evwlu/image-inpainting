import numpy as np
import tensorflow as tf

def dice_coefficient(true_images, completed_image):
    '''
    Dice coeffcient models how similar two images are, factoring in true positives,
    false positives, and false negatives; this is to say, this metric not only
    measures how many true positives are found, but also penalizes you for the
    number of false positives/negatives found (similar to precision). 
    '''
    true_f = np.array(tf.reshape(true_images, [-1]))
    complete_f = np.array(tf.reshape(completed_image, [-1]))

    # We can use "*" here to compute intersection with 
    moe = 0.1
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