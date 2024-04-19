import tensorflow as tf
import os


def load_and_count_images(anc_path, pos_path, neg_path):
    anchor = tf.data.Dataset.list_files(os.path.join(anc_path, '*.jpg')).take(300)
    positive = tf.data.Dataset.list_files(os.path.join(pos_path, '*.jpg')).take(300)
    negative = tf.data.Dataset.list_files(os.path.join(neg_path, '*.jpg')).take(300)

    print(f"Number of anchor images: {anchor.cardinality().numpy()}")
    print(f"Number of positive images: {positive.cardinality().numpy()}")
    print(f"Number of negative images: {negative.cardinality().numpy()}")
    return anchor, positive, negative


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


def pair_and_label_images(anchor, positive, negative):
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    return positives.concatenate(negatives)