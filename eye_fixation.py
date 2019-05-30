import os
import imageio
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from kld import kld


# Generator function will output one (image, target) tuple at a time,
# and shuffle the data for each new epoch
def data_generator(imgs, targets):
    while True:  # produce new epochs forever
        # Shuffle the data for this epoch
        idx = np.arange(imgs.shape[0])
        np.random.shuffle(idx)

        imgs = imgs[idx]
        targets = targets[idx]
        for i in range(imgs.shape[0]):
            yield imgs[i], targets[i]


def get_batch_from_generator(gen, batchsize):
    batch_imgs = []
    batch_fixations = []
    for i in range(batchsize):
        img, target = next(gen)
        batch_imgs.append(img)
        batch_fixations.append(target)
    return np.array(batch_imgs), np.array(batch_fixations)


def create_network_v2(imgs_normalized, is_training):
    vgg_weight_file = 'vgg16-conv-weights.npz'
    # Load VGG16 weights from file
    weights = np.load(vgg_weight_file)
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(initial_value=weights['conv1_1_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv1_1_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(imgs_normalized, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(initial_value=weights['conv1_2_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv1_2_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool1') as scope:
        pool = tf.layers.max_pooling2d(act, pool_size=(2, 2), strides=(2, 2), padding='same')

    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(initial_value=weights['conv2_1_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv2_1_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(initial_value=weights['conv2_2_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv2_2_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool2') as scope:
        pool = tf.layers.max_pooling2d(act, pool_size=(2, 2), strides=(2, 2), padding='same')

    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(initial_value=weights['conv3_1_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv3_1_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(initial_value=weights['conv3_2_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv3_2_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(initial_value=weights['conv3_3_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv3_3_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(act, pool_size=(2, 2), strides=(2, 2), padding='same')

    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(initial_value=weights['conv4_1_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv4_1_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(initial_value=weights['conv4_2_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv4_2_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(initial_value=weights['conv4_3_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv4_3_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('pool4') as scope:
        pool4 = tf.layers.max_pooling2d(act, pool_size=(2, 2), strides=(1, 1), padding='same')

    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(initial_value=weights['conv5_1_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv5_1_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(initial_value=weights['conv5_2_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv5_2_b'], trainable=False, name="biases")
        conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(initial_value=weights['conv5_3_W'], trainable=False, name="weights")
        biases = tf.Variable(initial_value=weights['conv5_3_b'], trainable=False, name="biases")
        conv5_3 = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv5_3, biases)
        act = tf.nn.relu(out, name=scope)

    concat_features = tf.concat([pool3, pool4, act], axis=3)
    # encoding network
    dropout = tf.layers.dropout(concat_features, rate=0.5, training=is_training)
    conv_en1 = tf.layers.conv2d(dropout, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv_en2 = tf.layers.conv2d(conv_en1, filters=1, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    return tf.image.resize_images(conv_en2, (180, 320), method=tf.image.ResizeMethod.BICUBIC)


# function for creating the network which does the job
def create_network(imgs_normalized, is_training):
    weights = np.load('vgg16-conv-weights.npz')
    # the feature extraction network
    conv1_1 = tf.layers.conv2d(imgs_normalized, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv1_1_W']),
                               bias_initializer=tf.constant_initializer(weights['conv1_1_b']), trainable=False)
    conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv1_2_W']),
                               bias_initializer=tf.constant_initializer(weights['conv1_2_b']), trainable=False)
    pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=2, padding="same")

    conv2_1 = tf.layers.conv2d(pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv2_1_W']),
                               bias_initializer=tf.constant_initializer(weights['conv2_1_b']),
                               trainable=False)
    conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv2_2_W']),
                               bias_initializer=tf.constant_initializer(weights['conv2_2_b']),
                               trainable=False)
    pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=[2, 2], strides=2, padding="same")

    conv3_1 = tf.layers.conv2d(pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv3_1_W']),
                               bias_initializer=tf.constant_initializer(weights['conv3_1_b']),
                               trainable=False)
    conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv3_2_W']),
                               bias_initializer=tf.constant_initializer(weights['conv3_2_b']),
                               trainable=False)
    conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv3_3_W']),
                               bias_initializer=tf.constant_initializer(weights['conv3_3_b']),
                               trainable=False)
    pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=[2, 2], strides=2, padding="same")

    conv4_1 = tf.layers.conv2d(pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv4_1_W']),
                               bias_initializer=tf.constant_initializer(weights['conv4_1_b']),
                               trainable=False)
    conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv4_2_W']),
                               bias_initializer=tf.constant_initializer(weights['conv4_2_b']),
                               trainable=False)
    conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv4_3_W']),
                               bias_initializer=tf.constant_initializer(weights['conv4_3_b']),
                               trainable=False)
    pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=[2, 2], strides=1, padding="same")

    conv5_1 = tf.layers.conv2d(pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv5_1_W']),
                               bias_initializer=tf.constant_initializer(weights['conv5_1_b']),
                               trainable=False)
    conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv5_2_W']),
                               bias_initializer=tf.constant_initializer(weights['conv5_2_b']),
                               trainable=False)
    conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                               kernel_initializer=tf.constant_initializer(weights['conv5_3_W']),
                               bias_initializer=tf.constant_initializer(weights['conv5_3_b']),
                               trainable=False)

    concat_features = tf.concat([pool3, pool4, conv5_3], axis=3)

    # encoding network
    dropout = tf.layers.dropout(concat_features, rate=0.5, training=is_training)
    conv_en1 = tf.layers.conv2d(dropout, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv_en2 = tf.layers.conv2d(conv_en1, filters=1, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    return tf.image.resize_images(conv_en2, (180, 320), method=tf.image.ResizeMethod.BICUBIC)


# function for creating data-set for training and validation data with their fixations
def create_dataset_from_images(folder):
    image_folder = os.path.join(folder, 'images')
    fixation_folder = os.path.join(folder, 'fixations')
    data = []
    fixations = []
    for filename in os.listdir(image_folder):
        image = imageio.imread(os.path.join(image_folder, filename))
        if image is not None:
            data.append(image)
    for filename in os.listdir(fixation_folder):
        fixation = imageio.imread(os.path.join(fixation_folder, filename))
        if fixation is not None:
            fixation = np.expand_dims(fixation, -1)  # adds singleton dimension so fixation size is (180,320,1)
            fixations.append(fixation)
    data = np.array(data)
    fixations = np.array(fixations)
    return data, fixations


def find_last_checkpoint_no():
    file = open("checkpoints/lastCheckpointNo.txt", "r")
    return int(file.read())


def save_last_checkpoint_no(number):
    file = open("checkpoints/lastCheckpointNo.txt", "w")
    file.write(str(number))


def run_validation(predictions, validation_fixations):
    sum = 0.0
    for i in range(len(validation_fixations)):
        sum += kld(validation_fixations[i], predictions[i])
    sum = sum / len(validation_fixations)
    print("Average KL-Divergence is " + str(sum))


# main function of the system where it starts processing
def main():
    print(tf.__version__)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # acquire the training images and fixations
    train_data, train_fixations = create_dataset_from_images('Data/train')
    # acquire the validation images and fixations
    validation_data, validation_fixations = create_dataset_from_images('Data/val')

    # Initialise Session
    # placeholders for images and fixation maps
    image = tf.placeholder(tf.uint8, (None, 180, 320, 3))
    fixation_map = tf.placeholder(tf.uint8, (None, 180, 320, 1))
    is_training = tf.placeholder(tf.bool, shape=[])     # is it training run or not
    # As preprocessing, the average RGB value must be subtracted.
    with tf.name_scope('preprocess') as scope:
        imgs = tf.image.convert_image_dtype(image, tf.float32) * 255.0
        fixations_normalized = tf.image.convert_image_dtype(fixation_map, tf.float32)  # convert fixations to float
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        imgs_normalized = imgs - mean
    # Init network
    network = create_network_v2(imgs_normalized, is_training)
    # Define loss function
    with tf.name_scope('loss') as scope:
        # normalize saliency
        max_value_per_image = tf.reduce_max(network, axis=[1, 2, 3], keepdims=True)
        predicted_saliency = (network / max_value_per_image)

        # Prediction is smaller than target, so downscale target to same size
        target_shape = predicted_saliency.shape[1:3]
        target_downscaled = tf.image.resize_images(fixations_normalized, target_shape)

        # Loss function from Cornia et al. (2016) [with higher weight for salient pixels]
        alpha = 1.01
        weights = 1.0 / (alpha - target_downscaled)
        loss = tf.losses.mean_squared_error(labels=target_downscaled,
                                            predictions=predicted_saliency,
                                            weights=weights)

        # Optimizer settings from Cornia et al. (2016) [except for decay]
        optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True)
        minimize_op = optimizer.minimize(loss)

    # Add a scalar summary for monitoring the loss
    loss_summary = tf.summary.scalar(name="loss", tensor=loss)

    # Number of batches and batch size
    num_batches = 10001
    batch_size = 16

    # for saving checkpoints
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        #writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        last_checkpoint = find_last_checkpoint_no()

        # restore session
        saver.restore(sess, "checkpoints/my-model-" + str(last_checkpoint))

        gen = data_generator(train_data, train_fixations)
        for b in range(last_checkpoint, num_batches):
            print('Starting batch {:d}'.format(b))
            batch_imgs, batch_fixations = get_batch_from_generator(gen, batch_size)
            _, l, batch_loss = sess.run([minimize_op, loss_summary, loss],
                                        feed_dict={image: batch_imgs,
                                                   fixation_map: batch_fixations, is_training: True})
            #writer.add_summary(l, global_step=b)
            if b % 100 == 0:
                print('Batch {:d} done: batch loss {:f}'.format(b, batch_loss))

            if b % 1000 == 0 and b != last_checkpoint:
                save_path = saver.save(sess, "checkpoints/my-model", global_step=b)
                save_last_checkpoint_no(b)

        predictions = np.zeros((len(validation_data), 180, 320, 3))

        for i in range(len(validation_data)):
            validation_image = np.zeros((1, 180, 320, 3))
            validation_image[0] = validation_data[i]
            saliency_maps = sess.run(predicted_saliency, feed_dict={image: validation_image, is_training: False})
            #upsampled_saliency = resize(saliency_maps[0], output_shape=(180, 320, 1))
            predictions[i] = saliency_maps[0]
            imageio.imwrite("predictions/" + str(i + 1201) + ".jpg", saliency_maps[0])


if __name__ == "__main__":
    main()
