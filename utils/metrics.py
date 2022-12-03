import numpy as np

def dice_coefficient(true_images, completed_image):
    '''
    Dice coeffcient models how similar two images are, factoring in true positives,
    false positives, and false negatives; this is to say, this metric not only
    measures how many true positives are found, but also penalizes you for the
    number of false positives/negatives found (similar to precision). 
    '''
    true_f = true_images.flatten()
    complete_f = completed_image.flatten()

    # We can use "*" here to compute intersection with 
    intersection = np.sum(true_f * complete_f)
    true_sum = np.sum(true_f)
    completed_sum = np.sum(complete_f)

    # We use epsilon in the rare event that there 
    epsilon = 0.0001

    dice_coeff = (2. * intersection + epsilon) / (np.sum(true_sum) + np.sum(completed_sum) + epsilon)
    return dice_coeff