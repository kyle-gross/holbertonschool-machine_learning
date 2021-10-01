#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt


def preprocess_data(X, Y):
    """Pre-processes the data for the model.

    Args:
        X (numpy.ndarray)(m,32,32,3): Contains the CIFAR 10 data
            m: number of data points
        Y (numpy.ndarray)(m,): Contains the CIFAR 10 labels for X

    Returns:
        X_p, Y_p
        X_p (numpy.ndarray): contains preprocessed X
        Y_p (numpy.ndarray): contains preprocessed Y
    """
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y


def load_dataset():
    """Loads and preprocesses the cifar10 dataset.

    Returns:
        x_train, y_train, x_test, y_test, respectively
    """
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    return x_train, y_train, x_test, y_test


def plot_model(history):
    """Plots the results of training."""
    plt.figure(1, figsize=(15, 8))

    plt.subplot(221)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])

    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])

    plt.show()


if __name__ == '__main__':
    # Load dataset
    X_train, Y_train, X_test, Y_test = load_dataset()

    # Load InceptionResNetv2 trained on imagenet dataset without classifier
    inception = K.applications.InceptionResNetV2(
        weights='imagenet',
        include_top=False
    )

    # Create lambda layer to resize images for InceptionResNetv2 (299 x 299)
    inputs = K.Input(shape=(32, 32, 3))
    resize = K.layers.Lambda(
        lambda image: tf.image.resize_images(image, (299, 299))
    )(inputs)

    # Create new classifier
    base = inception(resize, training=False)
    x = K.layers.GlobalAveragePooling2D()(base)
    x = K.layers.Dense(512, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    output = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs=inputs, outputs=output)

    # Freeze pre-trained layers
    inception.trainable = False
    # model.summary()

    # Compile model
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model on cifar10 dataset
    history = model.fit(
        X_train,
        Y_train,
        epochs=4,
        batch_size=32,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )

    # Re-compile
    # model.compile(
    #     optimizer=K.optimizers.Adam(),
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    # Save to current working directory as 'cifar10.h5'
    model.save('cifar10.h5')

    # Evaluate model
    # model.evaluate

    # Plot results
    # plot_model(history)
