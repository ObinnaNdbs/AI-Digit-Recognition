import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data loaded and normalized successfully!")

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a vector
    Dense(128, activation='relu'),  # Hidden layer
    Dropout(0.2),                   # Regularization to prevent overfitting
    Dense(10, activation='softmax') # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

print("Model training complete!")

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Save the model
model.save('mnist_classifier.h5')
print("Model saved successfully!")
