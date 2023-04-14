import tensorflow as tf
import pandas as pd
import random, pickle
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def create_model(dropout: float=0.2,
                 activation: str='relu',
                 kernel_initializer: str='',
                 learning_rate: float=0.01, **kwargs) -> tf.keras.Model:
    """
    Create a neural network model.

    Args:
        dropout (float, optional): Dropout rate to be applied during training. Defaults to 0.2.
        activation (str, optional): Activation function to be used. Defaults to 'relu'.
        kernel_initializer (str, optional): Kernel initialization technique. Defaults to ''.
        learning_rate (float, optional): Optimizer learning rate. Defaults to 0.01.

    Returns:
        tf.keras.Model: Compiled neural network model.
    """
    
    model = tf.keras.Sequential([
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(1024, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(512, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.Dense(7, activation='softmax', kernel_initializer=kernel_initializer)],
    name="Covertype_Model")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def find_best_hyperparameters(X_train, y_train, X_val, y_val, num_iterations: int=50) -> Dict[str, float]:
    """
    Find the best set of hyperparameters for the neural network model.

    Args:
        X_train (_type_): Training data.
        y_train (_type_): Training labels.
        X_val (_type_):  Validation data.
        y_val (_type_): Validation labels.
        num_iterations (int, optional): Number of iterations to perform. Defaults to 50.

    Returns:
        Dict[str, float]: Name and value of the best set of hyperparameters.
    """

    # Define the ranges for the hyperparameters
    dropout_range = [0.1, 0.2, 0.3]
    activation_range = ['relu', 'tanh']
    kernel_initializer_range = ['glorot_uniform', 'he_uniform', 'lecun_uniform']
    learning_rate_range = [0.001, 0.01, 0.005]
    batch_size_range = [256, 512, 1024]

    # Define the best set of hyperparameters and its performance
    best_hyperparameters = {}
    best_performance = float('-inf')

    # Perform the random search
    with open('./models/checkpoints/hyperparameter_logs.txt', 'w') as f:
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}")
            # Sample a set of hyperparameters
            dropout = random.choice(dropout_range)
            activation = random.choice(activation_range)
            kernel_initializer = random.choice(kernel_initializer_range)
            learning_rate = random.choice(learning_rate_range)
            batch_size = random.choice(batch_size_range)

            # Create a model with the sampled hyperparameters
            model = create_model(dropout=dropout, activation=activation, kernel_initializer=kernel_initializer, learning_rate=learning_rate)

            # Train the model on the training set and evaluate its performance on the validation set
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=5, validation_data=(X_val, y_val))
            performance = history.history['val_accuracy'][-1]

            # If the performance of the model with the sampled hyperparameters is better than the best performance so far, update the best set of hyperparameters
            if performance > best_performance:
                best_hyperparameters = {'dropout': dropout,
                                        'activation': activation,
                                        'kernel_initializer': kernel_initializer,
                                        'learning_rate': learning_rate,
                                        'batch_size': batch_size}
                best_performance = performance
                pr = f"Best performance --> {performance} for parameters:\n{best_hyperparameters}"
                f.write(pr+'\n')
                print(pr)

    # Return the best set of hyperparameters
    return best_hyperparameters


def train_nn(X_train: pd.DataFrame,
             y_train: pd.Series,
             X_test: pd.DataFrame, 
             y_test: pd.Series,
             hyperparmeters: Dict[str, float]) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    This function trains a neural network model and test the model accuracy.

    Args:
        data (pd.DataFrame): The data to train the model on.

    Returns:
        Tuple[tf.keras.Model, float]: The trained model and the accuracy of the model on unseen data.
    """
    assert len(X_train) == len(y_train), "X_train and y_train must be of the same length"
    assert len(X_test) == len(y_test), "X_test and y_test must be of the same length"

    #Define model
    model = create_model(**hyperparmeters)

    checkpoint = tf.keras.callbacks.ModelCheckpoint("./models/checkpoints/checkpoint_nn.h5",
                                                     monitor='val_accuracy',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max')

    #Train model
    history = model.fit(X_train,
                        y_train,
                        epochs=70,
                        batch_size=hyperparmeters['batch_size'],
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint])

    #Save model
    model.load_weights("./models/checkpoints/checkpoint_nn.h5")
    return model, history


def train_and_test_nn(data: pd.DataFrame) -> None:
    """
    This function trains a neural network model and test the model accuracy.

    Args:
        data (pd.DataFrame): The data to train the model on.

    """
    # Split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1],
                                                        tf.keras.utils.to_categorical(data.iloc[:, -1]-1, num_classes=7),
                                                        test_size=0.2, random_state=234)
    # Split the test set into validation and test set
    val_len = len(X_test)//2
    X_val, y_val = X_test[:val_len], y_test[:val_len]
    X_test, y_test = X_test[val_len:], y_test[val_len:]

    # Find the best hyperparameters
    best_hyperparameters = find_best_hyperparameters(X_train.to_numpy(), y_train, X_val.to_numpy(), y_val, 20)

    # Save the best hyperparameters
    with open('./models/checkpoints/best_hyperparameters.pkl', 'wb') as f:
        pickle.dump(best_hyperparameters, f)
    
    # Train the model with the best hyperparameters
    nn, history = train_nn(X_train.to_numpy(), y_train, X_val.to_numpy(), y_val, best_hyperparameters)


    plot_learning_curve(history)
    # Evaluate the model on the test set
    acc = nn.evaluate(X_test.to_numpy(), y_test)[1]
    print(f"Test accuracy: {acc}")

    nn.save("./models/checkpoints/neural_net.h5")
    
    matrix = confusion_matrix(y_test.argmax(axis=-1), nn.predict(X_test.to_numpy()).argmax(axis=-1))

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                  display_labels=[i + 1 for i in range(7)])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.title(f"Neural Network confusion matrix\nTest accuracy: {acc:.4f}")
    plt.savefig(f'./models/evaluation/nerual_net.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_learning_curve(history: tf.keras.callbacks.History) -> None:
    """
    This function plots the learning curve of a model.

    Args:
        history (tf.keras.callbacks.History): The history of the model.
        model_name (str): The name of the model.
    """
    # Plot the learning curve
    fig_acc, ax_acc = plt.subplots(figsize=(11, 7), dpi=300)
    ax_acc.plot(history.history['accuracy'], label='train_accuracy')
    ax_acc.plot(history.history['val_accuracy'], label='val_accuracy')
    ax_acc.set_title('Learnin Curve')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim([0.5, 1])
    ax_acc.legend(loc='lower right')
    fig_acc.savefig('./models/evaluation/accuracy_curve.png', dpi=300, bbox_inches='tight')

    # create the plot for loss
    fig_loss, ax_loss = plt.subplots(figsize=(11, 7), dpi=300)
    ax_loss.plot(history.history['loss'], label='loss')
    ax_loss.plot(history.history['val_loss'], label='val_loss')
    ax_loss.set_title('Loss Curve')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='lower right')
    fig_loss.savefig('./models/evaluation/loss_curve.png', dpi=300, bbox_inches='tight')