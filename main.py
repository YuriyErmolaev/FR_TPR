from main_code.setup import setup_gpu, setup_paths
setup_gpu()
#step 1 # Setup GPU and data paths # setup.py # move into top for avoid errors
from main_code.data_loading import load_and_count_images, pair_and_label_images  #preprocess use in data pipeline
from main_code.data_pipeline import configure_training_pipeline, split_data
from main_code.model_architecture import make_embedding, make_siamese_model
from main_code.training import train

POS_PATH, NEG_PATH, ANC_PATH = setup_paths()

# Load dataset images, preprocess, and label pairs for training # data_loading.py
anchor, positive, negative = load_and_count_images(ANC_PATH, POS_PATH, NEG_PATH)
data = pair_and_label_images(anchor, positive, negative)

# Preprocess paired images, configure the training pipeline, and split data sets # data_pipeline.py
data = configure_training_pipeline(data)
train_data, test_data = split_data(data)

# Init and show the emb model
embedding = make_embedding()
embedding.summary()

# Init and show Siam network
siamese_model = make_siamese_model()
siamese_model.summary()

# Exec train with number epochs
EPOCHS = 50
train(train_data, siamese_model, EPOCHS)

#save model to file
siamese_model.save('siamesemodelv5.keras')