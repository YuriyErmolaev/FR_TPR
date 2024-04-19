from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import os
from setup import setup_gpu, setup_paths
from data_loading import load_and_count_images, pair_and_label_images, preprocess #preprocess use in data pipeline
from data_pipeline import configure_training_pipeline, split_data
from model_architecture import make_embedding, make_siamese_model

#step 1
# Setup GPU and data paths
# setup.py
setup_gpu()
POS_PATH, NEG_PATH, ANC_PATH = setup_paths()


# Step 2, 3, 4
# Load dataset images, preprocess, and label pairs for training
# data_loading.py
anchor, positive, negative = load_and_count_images(ANC_PATH, POS_PATH, NEG_PATH)
data = pair_and_label_images(anchor, positive, negative)


# Step 5 to 8
# Preprocess paired images, configure the training pipeline, and split data sets
# data_pipeline.py
data = configure_training_pipeline(data)
train_data, test_data = split_data(data)

# step 9 # Def emb model arch # model_architecture.py

# step 10
# Init and show the emb model
embedding = make_embedding()
embedding.summary()

#step 11 # Def custom layer to calc L1 distance for Siamese network # model_architecture.py
#step 12 # def Construct Siamese network model # model_architecture.py


#step 13
# Init and show Siam network
siamese_model = make_siamese_model()
siamese_model.summary()


#step 14
# Set insts loss_funct and optimizer for the Siamese network

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


#step 15
# Set checkpoint mechanism for training

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)



#step 16
# Def train step for Siamese model


@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


#step 17
# Def train process over epochs

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx+1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


#step 18
# Exec train with number epochs

EPOCHS = 50
train(train_data, EPOCHS)


#step 19
#save model to file


siamese_model.save('siamesemodelv3.keras')