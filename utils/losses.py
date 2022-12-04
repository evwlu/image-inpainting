import tensorflow as tf

def completion_loss(true_image, completed_image):
    l2 = tf.nn.l2_loss(true_image - completed_image)
    mean = tf.reduce_mean(l2)

    return mean

def discriminator_loss(real_pred, fake_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = bce(tf.ones(real_pred.shape), real_pred)
    fake_loss = bce(tf.zeros(fake_pred.shape), fake_pred)

    # note: make sure to use tf.add instead of '+'!
    return tf.add(real_loss, fake_loss)

def joint_loss(true_image, completed_image, real_pred, fake_pred, alpha):
    gen_loss = completion_loss(true_image, completed_image)
    disc_loss = alpha * (discriminator_loss(real_pred, fake_pred))

    return gen_loss + disc_loss
