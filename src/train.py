from pathlib import Path
import logging

import tensorflow as tf
from keras import layers, models, optimizers, losses
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')


class Config:
    INPUT_SHAPE = (256, 256, 3)
    TARGET_SIZE = (256, 256)
    BATCH_SIZE = 32

    TRAIN_DATA_PATH = Path().cwd() / 'data' / 'Training'
    TEST_DATA_PATH = Path().cwd() / 'data' / 'Testing'
    MODEL_PATH = 'inception_v3_brain_tumor_classifier.h5'
    TFLITE_MODEL_PATH = 'inception_v3_brain_tumor_classifier.tflite'

    LEARNING_RATE = 0.01
    INNER_SIZE = 128
    DROPRATE = 0.0
    EPOCHS = 20


def create_image_data_loader(
        directory_path,
        target_size=None,
        batch_size=None,
        preprocessing_function=None,
        shuffle=True
    ):

    target_size = target_size or Config.TARGET_SIZE
    batch_size = batch_size or Config.BATCH_SIZE

    data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)
    data_loader = data_generator.flow_from_directory(
        directory=directory_path,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader


def create_base_model():
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=Config.INPUT_SHAPE
    )
    base_model.trainable = False
    return base_model


def build_custom_layers(inputs, inner_size, droprate):
    base = create_base_model()(inputs, training=False)
    vectors = layers.GlobalAveragePooling2D()(base)
    inner = layers.Dense(inner_size, activation='relu')(vectors)
    drop = layers.Dropout(droprate)(inner)
    return drop


def build_full_model(inner_size=None, droprate=None):
    inner_size = inner_size or Config.INNER_SIZE
    droprate = droprate or Config.DROPRATE

    inputs = layers.Input(shape=Config.INPUT_SHAPE)
    custom_layers = build_custom_layers(inputs, inner_size, droprate)
    outputs = layers.Dense(4)(custom_layers)
    model = models.Model(inputs, outputs)
    return model


def compile_model(model, learning_rate=None):
    learning_rate = learning_rate or Config.LEARNING_RATE

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss = losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def make_model(learning_rate=None, inner_size=None, droprate=None):
    learning_rate = learning_rate or Config.LEARNING_RATE
    inner_size = inner_size or Config.INNER_SIZE
    droprate = droprate or Config.DROPRATE

    custom_model = build_full_model(inner_size=inner_size, droprate=droprate)
    compiled_model = compile_model(custom_model, learning_rate=learning_rate)
    return compiled_model


def train_model(model, train_data_loader, test_data_loader, epochs=None, callbacks=None):
    epochs = epochs or Config.EPOCHS
    model.fit(
        train_data_loader,
        epochs=epochs,
        validation_data=test_data_loader,
        callbacks=callbacks,
        steps_per_epoch=len(train_data_loader),
        validation_steps=len(test_data_loader)
    )


def convert_to_tflite(model_path, tflite_path):
    model = models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)


def main():
    logger.info('Creating training data loader...')
    train_data_loader = create_image_data_loader(
        Config.TRAIN_DATA_PATH,
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE,
        preprocessing_function=preprocess_input,
        shuffle=True
    )
    logger.info('Trining data loader created.')

    logger.info('Creating testing data loader...')
    test_data_loader = create_image_data_loader(
        Config.TEST_DATA_PATH,
        target_size=Config.TARGET_SIZE,
        batch_size=Config.BATCH_SIZE,
        preprocessing_function=preprocess_input,
        shuffle=False
    )
    logger.info('Testing data loader created.')

    logger.info('Building the model...')
    model = make_model(
        learning_rate=Config.LEARNING_RATE,
        inner_size=Config.INNER_SIZE,
        droprate=Config.DROPRATE
    )
    logger.info('Model created.')

    checkpoint_callback = ModelCheckpoint(
        filepath=Config.MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    logger.info('Training the model...')
    train_model(
        model=model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        callbacks=[checkpoint_callback]
    )
    logger.info('Model training is complete.')

    logger.info('Saving the model into TFLite format...')
    convert_to_tflite(Config.MODEL_PATH, Config.TFLITE_MODEL_PATH)
    logger.info('Model saved.')


if __name__ == '__main__':
    main()
