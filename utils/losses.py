import tensorflow as tf

def completion_loss(true_image, completed_image):
    l2 = tf.nn.l2_loss(completed_image - true_image)
    mean = tf.reduce_mean(l2)

    return mean

def discriminator_loss(real_pred, fake_pred):
    bce = tf.keras.losses.BinaryCrossentropy()

    real_loss = bce(tf.ones(real_pred.shape), real_pred)
    fake_loss = bce(tf.zeros(fake_pred.shape), fake_pred)

    # note: make sure to use tf.add instead of '+'!
    return tf.add(real_loss, fake_loss)

def joint_loss(true_image, completed_image, fake_pred, alpha=0.0004):
    gen_loss = completion_loss(true_image, completed_image)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    disc_loss = alpha * bce(tf.ones(fake_pred.shape), fake_pred)

    return gen_loss + disc_loss
