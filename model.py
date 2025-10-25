# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import the specific module from tensorflow.compat.v1
from tensorflow.compat.v1 import ragged

# ...

# Use the updated module for RaggedTensorValue
# Instead of: tf.ragged.RaggedTensorValue
# Use: ragged.RaggedTensorValue


batch_size = 32

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'dataset',  # This is the source directory for training images
    target_size=(200, 200),  # All images will be resized to 200 x 200
    batch_size=batch_size,
    # Specify the classes explicitly
    classes=['MildDemented', 'ModerateDemented','NonDemented','VeryMildDemented'],
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode='categorical')

import tensorflow as tf

model = Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),
    # The second convolution
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # The third convolution
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # The fourth convolution
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # The fifth convolution
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # Flatten the results to feed into a dense layer
    Flatten(),
    # 128 neurons in the fully-connected layer
    Dense(128, activation='relu'),
    # 4 output neurons for 4 classes with softmax activation
    Dense(4, activation='softmax')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# Define the number of epochs
n_epochs = 30

# Use 'validation_data' to monitor validation accuracy and loss during training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=n_epochs,
    verbose=1)

# Plot training accuracy
plt.plot(history.history['acc'])
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Save the model
model.save('model.h5')
