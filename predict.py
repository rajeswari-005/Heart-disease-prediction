import tensorflow as tf 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
classifierLoad = tf.keras.models.load_model('model.h5')

# Load the sample image
test_image = cv2.imread('1.jpg')
test_image = cv2.resize(test_image, (200, 200))  # Resize the image
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Predict the class
result = classifierLoad.predict(test_image)

# Get the predicted class index (assuming softmax output)
predicted_class = np.argmax(result)

# Initialize recommendation variable
recommendation = ""
medicine = ""

# Handle the predicted class
if predicted_class == 0:
    print("Mild Demented")
    recommendation = "Focus on memory exercises and a healthy diet."
    medicine = "Donepezil or Memantine, based on doctor's recommendation."
    
elif predicted_class == 1:
    print("Moderate Demented")
    recommendation = "Seek cognitive therapy and monitor behavior closely."
    medicine = "Galantamine or Rivastigmine, based on doctor's prescription."
    
elif predicted_class == 2:
    print("Non-Demented")
    recommendation = "Maintain regular check-ups and a healthy lifestyle."
    medicine = "No medication necessary at this stage."
    
elif predicted_class == 3:
    print("Very Mild Demented")
    recommendation = "Encourage social engagement and light mental exercises."
    medicine = "Consider Donepezil, with doctor's approval."

# Print the result and recommendation
print(f"Prediction: {['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented'][predicted_class]}")
print("Recommendation:", recommendation)
print("Medicine:", medicine)

# Display the image
plt.imshow(cv2.cvtColor(test_image.squeeze(), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Create a DataFrame with the recommendation and medicine
df = pd.DataFrame({'Prediction': [predicted_class],
                   'Recommendation': [recommendation],
                   'Medicine': [medicine]})

# Save the DataFrame to an Excel file
df.to_excel('recommendation.xlsx', index=False)
