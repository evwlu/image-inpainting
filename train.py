import argparse
import tensorflow as tf
from inpainter import ImageInpaint

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
    model = tf.keras.models.load_model(checkpoint)

    from functools import partial
    model.test    = partial(ImageInpaint.test,    model)
    model.train   = partial(ImageInpaint.train,   model)
    model.compile = partial(ImageInpaint.compile, model)

def train(model, train_images, batch_size, T_C, T_D, T):

    # note: apply data augmentation before training
    try:
        model.train(train_images, batch_size, T_C, T_D, T)
    except KeyboardInterrupt as e:
        print("Key-value interruption")

def test(model, test_images):

    # initialize stats, test the model, print out the stats at the end
    pass

if __name__ == '__main__':
    T_C, T_D, T = 1000, 600, 3600
    train_images, _ = get_data()
    model = ImageInpaint()
    compile_model(model)
    train(model, train_images, 25, 2500, 1000, 2000)

    '''
    args = parse_args()
    test_images = None
    if (args.task == 'train' or args.task == 'both'):
        train_images, test_images = get_data()
        model = ImageInpaint()
        
        compile_model(model)
        train(model, train_images)

        if (args.checkpoint_path):
            tf.keras.models.save_model(model, args.checkpoint_path)
    
    if (args.task == 'test' or args.task == 'both'):
        if (args.checkpoint_path):
            if (test_images == None):
                _, test_images = get_data()
            
            model = load_model(args.checkpoint_path)
            test(model, test_images)
    '''