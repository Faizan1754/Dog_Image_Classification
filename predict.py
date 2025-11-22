import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class DogClassifier:
    def __init__(self, model_path, img_size=(224, 224)):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def predict(self, image_path, show_image=True):
        """
        Predict if image contains a dog - WITH CORRECT LABEL MAPPING
        """
        processed_image = self.preprocess_image(image_path)
        
        if processed_image is None:
            return {"error": "Could not process image"}
        
        # Make prediction - model outputs probability of class 1 (not_dogs)
        raw_prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        print(f"Raw model output (probability of not_dogs): {raw_prediction:.4f}")
        
        # CORRECT INTERPRETATION:
        # raw_prediction = probability of not_dogs (class 1)
        # So: dog_probability = 1 - raw_prediction
        dog_probability = 1 - raw_prediction
        not_dog_probability = raw_prediction
        
        # Determine class
        if dog_probability > 0.5:
            predicted_class = 'Yes, this is a dog'
            confidence = dog_probability
        else:
            predicted_class = 'This is not a dog'
            confidence = not_dog_probability
        
        print(f"Dog probability: {dog_probability:.4f}")
        print(f"Not-dog probability: {not_dog_probability:.4f}")
        
        # Display image if requested
        if show_image:
            self._display_prediction(image_path, predicted_class, confidence, dog_probability)
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'dog_probability': float(dog_probability),
            'not_dog_probability': float(not_dog_probability),
            'raw_output': float(raw_prediction)
        }
    
    def _display_prediction(self, image_path, predicted_class, confidence, dog_probability):
        """Display the image with prediction results"""
        img = Image.open(image_path)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Prediction: {predicted_class}\n'
                 f'Confidence: {confidence:.4f}\n'
                 f'Dog Probability: {dog_probability:.4f}')
        plt.axis('off')
        plt.show()
    
    def test_with_correction(self, image_path):
        """Test prediction with correction"""
        result = self.predict(image_path)
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Dog Probability: {result['dog_probability']:.4f}")
        print(f"Not-Dog Probability: {result['not_dog_probability']:.4f}")
        
        return result