# trainmodel.py
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Set your dataset folders
base_dir = r"C:\Users\nanda\OneDrive\Documents\Nanda's ML Tasks\SVM\PetImages"
cat_folder = os.path.join(base_dir, "Cat")
dog_folder = os.path.join(base_dir, "Dog")

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img.flatten())
                labels.append(label)
        except Exception as e:
            print(f"Skipping file {img_path}: {e}")
    return images, labels

# Load images
cat_images, cat_labels = load_images_from_folder(cat_folder, 0)  # 0 = Cat
dog_images, dog_labels = load_images_from_folder(dog_folder, 1)  # 1 = Dog

# Combine datasets
X = cat_images + dog_images
y = cat_labels + dog_labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model
model_path = os.path.join(base_dir, 'cat_dog_svm_model.pkl')
joblib.dump(clf, model_path)
print(f"✅ Model saved at {model_path}")
