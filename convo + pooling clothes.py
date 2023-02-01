import tensorflow as tf

class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

callback = Mycallback()
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels),(test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""model.compile(optimizer="adam",loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images, training_labels, epochs=50, callbacks=[callback])

model.evaluate(test_images,test_labels)"""
print(model.summary())