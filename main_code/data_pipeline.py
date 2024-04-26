from main_code.data_loading import preprocess


def preprocess_twin(input_img, validation_img, label):
    # Define preprocessing for paired images
    return (preprocess(input_img), preprocess(validation_img), label)


def configure_training_pipeline(data):
    # Configure the data loader pipeline for training
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)
    return data


def split_data(data):
    # Split data into training and testing sets
    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return train_data, test_data
