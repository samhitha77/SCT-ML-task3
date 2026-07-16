
# Cat vs Dog Classification using SVM
# SkillCraft Technology - Task 03

import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -------------------------------
# Dataset Path
# -------------------------------

DATASET_PATH = "dataset"

IMAGE_SIZE = (128, 128)

images = []
labels = []

# -------------------------------
# HOG Feature Extraction Function
# -------------------------------

def extract_hog(image):

    image = cv2.resize(image, IMAGE_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    return feature

# -------------------------------
# Load Images
# -------------------------------

categories = ["Cat", "Dog"]

MAX_IMAGES = 1000

for label, category in enumerate(categories):

    folder = os.path.join(DATASET_PATH, category)

    print(f"\nLoading {category} images...")

    image_list = os.listdir(folder)[:MAX_IMAGES]

    for image_name in tqdm(image_list):

        path = os.path.join(folder, image_name)

        image = cv2.imread(path)

        if image is None:
            continue

        try:
            feature = extract_hog(image)
            images.append(feature)
            labels.append(label)

        except:
            pass

print("\nDataset Loaded Successfully!")

X = np.array(images)
y = np.array(labels)

print("Unique Labels:", np.unique(y))
print("Cat images:", np.sum(y == 0))
print("Dog images:", np.sum(y == 1))

print("Total Images:", len(X))

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Images :", len(X_train))
print("Testing Images :", len(X_test))
# ==========================================
# Train SVM Model
# ==========================================

print("\nTraining SVM Model...")

model = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(f"Accuracy : {accuracy*100:.2f}%")
print("===================================\n")

print("Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Cat", "Dog"]
))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

disp.plot(cmap="Blues")

os.makedirs("images", exist_ok=True)

plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "svm_model.pkl")


print("\nModel saved as svm_model.pkl")
# ==========================================
# Predict External Image
# ==========================================

print("\n========== Cat vs Dog Prediction ==========")

image_path = input("Enter image path: ").strip().strip('"')
print("Path entered:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("Invalid image path!")
else:

    feature = extract_hog(image)

    feature = feature.reshape(1, -1)

    prediction = model.predict(feature)[0]

    probability = model.predict_proba(feature)[0]

    confidence = np.max(probability) * 100

    label = "Cat 🐱" if prediction == 0 else "Dog 🐶"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title(f"{label}\nConfidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()

    print("\nPrediction :", label)
    print(f"Confidence : {confidence:.2f}%")