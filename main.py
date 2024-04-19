from setup import setup_gpu, setup_paths
setup_gpu()
#step 1 # Setup GPU and data paths # setup.py # move into top for avoid errors
from data_loading import load_and_count_images, pair_and_label_images, preprocess #preprocess use in data pipeline
from data_pipeline import configure_training_pipeline, split_data
from model_architecture import make_embedding, make_siamese_model
from training import train

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

# step 10 # Init and show the emb model
embedding = make_embedding()
embedding.summary()

#step 11 # Def custom layer to calc L1 distance for Siamese network # model_architecture.py
#step 12 # def Construct Siamese network model # model_architecture.py

#step 13
# Init and show Siam network
siamese_model = make_siamese_model()
siamese_model.summary()

#step 14 # Set insts loss_funct and optimizer for the Siamese network # training.py
#step 15 # Set checkpoint mechanism for training # training.py
#step 16 # Def train step for Siamese model # training.py
#step 17 # Def train process over epochs # training.py

#step 18
# Exec train with number epochs
EPOCHS = 50
train(train_data, siamese_model, EPOCHS)

#step 19
#save model to file
siamese_model.save('siamesemodelv3.keras')