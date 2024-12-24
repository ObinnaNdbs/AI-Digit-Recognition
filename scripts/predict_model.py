from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('mnist_classifier.h5')

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (optional, for dark backgrounds)
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array.reshape(1, 28, 28)  # Add batch dimension

# Test each uploaded image
image_paths = ['IMG_0097.jpg', 'IMG_0098.jpg', 'IMG_0099.jpg', 'IMG_0100.jpg', 'IMG_0101.jpg', 'IMG_0102.jpg']
for path in image_paths:
    processed_image = preprocess_image(path)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    print(f"Predicted digit for {path}: {predicted_label}")
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.show()
