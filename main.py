import numpy as np
import pandas as pd
import cv2
import os
import pathlib
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA

#path = "C:/Users/rupaa/Desktop/archive/Indian-monuments/images"
train_dir = "C:/Users/rupaa/Desktop/archive/Indian-monuments/images/train"
validation_dir = "C:/Users/rupaa/Desktop/archive/Indian-monuments/images/test"

# Get all the class names
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
val_data_dir= pathlib.Path(validation_dir)
val_class_names= np.array(sorted([item.name for item in val_data_dir.glob("*")]))

def load_data(target_dir, target_class, num_images):
    images = []
    labels = []
    target_folder = os.path.join(target_dir, target_class)
    image_files = os.listdir(target_folder)
    random.shuffle(image_files)  # Shuffle the image files
    for image_file in image_files[:num_images]:
        image_path = os.path.join(target_folder, image_file)
        if not os.path.isfile(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        images.append(image)
        labels.append(target_class)
    return images, labels


def load_validation_data(target_dir, target_class, num_images):
    images = []
    labels = []
    target_folder = os.path.join(target_dir, target_class)
    image_files = os.listdir(target_folder)
    random.shuffle(image_files)  # Shuffle the image files
    for image_file in image_files[:num_images]:
        image_path = os.path.join(target_folder, image_file)
        if not os.path.isfile(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        images.append(image)
        labels.append(target_class)
    return images, labels


image_size = (224, 224)

data = []
labels = []

num_images_per_class = 500  # Adjust this value as per your memory constraints

# Load training data
for class_name in class_names:
    images, class_labels = load_data(train_dir, class_name, num_images_per_class)
    data.extend(images)
    labels.extend(class_labels)

# Load validation data
for class_name in val_class_names:
    images, class_labels = load_validation_data(validation_dir,class_name, num_images_per_class)
    data.extend(images)
    labels.extend(class_labels)

df = pd.DataFrame({'Image': data, 'Label': labels})
# Flatten the images
df['Flattened_Image'] = df['Image'].apply(lambda x: x.flatten())

# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(df['Flattened_Image'], df['Label'], test_size=0.2, random_state=7)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=100)  # Adjust the number of components as per your requirement
X_train_pca = pca.fit_transform(np.stack(X_train))
X_test_pca = pca.transform(np.stack(X_test))

# Save the PCA model
joblib.dump(pca, 'pca_model.pkl')

rf = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]

parameters = {
    'n_estimators': n_estimators,
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10, 15, 30, 50],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [42],
    'bootstrap': [True, False]
}

clf = RandomizedSearchCV(estimator=rf, param_distributions=parameters, cv=10, verbose=2, n_jobs=4)
clf.fit(X_train_pca, y_train)
joblib.dump(clf, 'random_search_model.pkl')

print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
print(clf.best_estimator_)
print(f'Train accuracy: {clf.score(X_train_pca, y_train):.3f}')
print(f'Test accuracy: {clf.score(X_test_pca, y_test):.3f}')


def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    flattened_image = image.flatten()
    pca_image = pca.transform([flattened_image])
    predicted_label = clf.predict(pca_image)[0]
    return predicted_label


def plot_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Take input image path from the user
n = 1
while n == 1:
    image_path = input("Enter the path of the image: ")
    predicted_label = predict_image(image_path)
    # Plot the image
    plot_image(image_path)
    print("Predicted Label:", predicted_label)
    n = int(input("To continue press 1, else press 0: "))
