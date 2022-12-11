import argparse
import tensorflow as tf
import numpy as np
from inpainter import ImageInpaint
from completion_net import CompletionNetwork
from gan import LocalDiscriminator, GlobalDiscriminator

def parse_args():
    parser = argparse.ArgumentParser(description="Arg Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', required=True, choices=['train', 'test', 'both'], help='training or testing (a saved model)')
    parser.add_argument('--checkpoint_path', default='', help='saved model location')
    
    return parser.parse_args()

def get_data():
    (training_images, _), (test_images, _) = tf.keras.datasets.cifar100.load_data()
    return training_images, test_images

def compile_model(model):
    from utils.losses import completion_loss, discriminator_loss, joint_loss

    optimizer = tf.keras.optimizers.Adam()
    losses = [tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM), discriminator_loss, joint_loss]
    acc = None # fill in later with dice coeff

    model.compile(
        optimizer = optimizer,
        losses = losses,
        metrics = [acc]
    )

def load_model(checkpoint):
    model = tf.keras.models.load_model(checkpoint, 
        custom_objects = dict(
            CompletionNetwork = CompletionNetwork,
            LocalDiscriminator = LocalDiscriminator,
            GlobalDiscriminator = GlobalDiscriminator
        ),
        compile = False)

    from functools import partial
    model.test    = partial(ImageInpaint.test,    model)
    model.train   = partial(ImageInpaint.train,   model)
    model.compile = partial(ImageInpaint.compile, model)
    model.update_variables = partial(ImageInpaint.update_variables, model)
    
    return model

def train(model, train_images, batch_size, T_C, T_D, T, restore):

    # note: apply data augmentation before training
    try:
        augment_fn = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomContrast(0.10)
        ])
        model.train(train_images, batch_size, T_C, T_D, T, augment_fn, restore=restore)
    except KeyboardInterrupt as e:
        print("\nKey-value interruption")

def test(model, test_images):
    model.test()
    # initialize stats, test the model, print out the stats at the end
    pass

if __name__ == '__main__':

    # for now you can just run train.py w/o proving any command line arguments
    batch_size = 25
    #T_C, T_D, T = 1800, 200, 12000
    T_C, T_D, T = 1800, 200, 3000

    args = parse_args()
    test_images = None

    # for training, add `--task train --checkpoint_path ./model`
    if (args.task == 'train' or args.task == 'both'):
        train_images, test_images = get_data()
        np.random.shuffle(train_images)
        model = ImageInpaint()
        
        compile_model(model)
        train(model, train_images, batch_size, T_C, T_D, T, True)
        
        if (args.checkpoint_path):
            model(tf.random.normal((1,32,32,3)))
            tf.keras.models.save_model(model, args.checkpoint_path)
    
    if (args.task == 'test' or args.task == 'both'):
        if (args.checkpoint_path):
            if (test_images == None):
                _, test_images = get_data()

            model = load_model(args.checkpoint_path)
            test(model, test_images)
        else:
            print("ERROR: Need to provide checkpoint path!")