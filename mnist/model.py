import argparse
import tensorflow as tf

N_DIGITS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 5

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/tmp/data/', help='Path of training data.')
    parser.add_argument('--model-file', type=str, default='./mnist_model.keras', help='File to save the model.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate for training.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for training.')
    return parser.parse_args()

def load_data():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(N_DIGITS, activation='softmax')
    ])
    return model

def main():
    args = parse_arguments()
    (train_images, train_labels), (test_images, test_labels) = load_data()

    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(test_images, test_labels))

    model.save(args.model_file)
    print(f"Model saved to {args.model_file}")

if __name__ == '__main__':
    main()
