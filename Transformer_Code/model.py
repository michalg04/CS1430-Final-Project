# Based on keras tutorial for ViT: https://keras.io/examples/vision/image_classification_with_vision_transformer/

from curses import pair_content
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
from keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
import hyperparameters as hp
from preprocess import get_data
import argparse

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=tf.expand_dims(images, axis=3),
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection": self.projection,
            "position_embedding": self.position_embedding,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def display_patches(x_train):
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(hp.image_size, hp.image_size)
    )
    patches = Patches(hp.patch_size)(resized_image)
    print(f"Image size: {hp.image_size} X {hp.image_size}")
    print(f"Patch size: {hp.patch_size} X {hp.patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (hp.patch_size, hp.patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(x):
    inputs = layers.Input(shape=hp.input_shape)

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Create patches.
    patches = Patches(hp.patch_size)(augmented)

    # Encode patches.
    encoded_patches = PatchEncoder(hp.num_patches, hp.projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(hp.transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=hp.num_heads, key_dim=hp.projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=hp.transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=hp.mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    # logits = layers.Dense(hp.num_classes)(features)
    outputs = layers.Dense(hp.num_classes, activation="softmax")(features)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs) # logits

    return model

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def run_experiment(model, x_train, y_train, x_test, y_test):
    optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.learning_rate)

    # One hot labels
    y_train = tf.one_hot(y_train, 2)
    y_test = tf.one_hot(y_test, 2)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = "checkpoints/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        # save_best_only=True,
        # save_weights_only=True,
    )

    csv_logger = CSVLogger('log.csv', append=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min", restore_best_weights=True)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=hp.batch_size,
        epochs=hp.num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            csv_logger,
            early_stopping,
        ],
    )

    # model.load_weights(checkpoint_filepath)
    # _, accuracy = model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history

def run_eval(model, x_test, y_test, checkpoint):
    optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.learning_rate)

    y_test = tf.one_hot(y_test, 2)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            f1_metric,
        ],
    )

    model.load_weights(checkpoint)
    _, accuracy, recall, precision, f1 = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

if __name__ == '__main__':

    # Read in data
    # x_train, y_train, x_val, y_val, x_test, y_test = get_data("../../data/chest_xray/train", "../../data/chest_xray/val", "../../data/chest_xray/test")
    # x_train, y_train = get_data("../../data/chest_xray/train")
    x_test, y_test = get_data("../data/chest_xray/test")

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=False, type=bool)
    parser.add_argument("--load-checkpoint", default=None)
    args = parser.parse_args() 

    # vit_classifier = create_vit_classifier(x_train)

    if not args.evaluate:
        x_train, y_train = get_data("../data/chest_xray/train")
        vit_classifier = create_vit_classifier(x_train)
        history = run_experiment(vit_classifier, x_train, y_train, x_test, y_test)
    else:
        vit_classifier = create_vit_classifier(x_test)
        run_eval(vit_classifier, x_test, y_test, args.load_checkpoint)
