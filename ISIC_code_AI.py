import os

project_dir = os.path.join(os.sep, 'projectnb', 'cs640grp', 'materials', 'ISIC-2024_CS640')
os.listdir(project_dir)

import pandas

df_train = pandas.read_csv(os.path.join(project_dir, "train_metadata.csv"))
df_train

import matplotlib.pyplot as plt
import matplotlib.image as img

fig, axes = plt.subplots(1, 4, figsize = (10, 20))
for i in range(4):
    id = str(df_train["id"][i])
    image = img.imread(os.path.join(project_dir, "train_image", id + ".jpg"))
    axes[i].imshow(image)
    axes[i].set_title(id + ".jpg")
    axes[i].set_axis_off()
plt.show()

df_test = pandas.read_csv(os.path.join(project_dir, "test_metadata.csv"))
df_test

df_test.drop(columns = ['target'], inplace=True)

import pandas as pd
import numpy as np
import cv2
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from sklearn.preprocessing import FunctionTransformer
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,Concatenate
from sklearn.metrics import classification_report
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, save_img
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

dist = df_train.target.value_counts()

plt.figure(figsize=(4, 4))
plt.pie(dist.values, labels=dist.index.astype(str), autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Value Counts of Target Classes')
plt.show()

df_train.drop(columns = ['tbp_tile_type','tbp_lv_location'], inplace=True)

df_train['sex'] = df_train['sex'].fillna('Unknown')

from sklearn.preprocessing import OneHotEncoder

categorical_columns = ['anatom_site_general', 'tbp_lv_location_simple','sex']

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df_train[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([df_train, one_hot_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(categorical_columns, axis=1)

filtered_rows = df_encoded[(df_encoded['anatom_site_general_nan']==True) & (df_encoded['target']==0)]
filtered_rows.index

df_encoded = df_encoded.drop(filtered_rows.index)
df_encoded

df_encoded.drop(columns = ['tbp_lv_location_simple_Unknown','anatom_site_general_nan'], inplace=True)

df_encoded['age_approx'] = df_encoded['age_approx'].fillna(df_encoded['age_approx'].mean())

df_test['sex'] = df_test['sex'].fillna('Unknown')

one_hot_encoded = encoder.fit_transform(df_test[categorical_columns])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_test_encoded = pd.concat([df_test, one_hot_df], axis=1)

df_test_encoded = df_test_encoded.drop(categorical_columns, axis=1)

df_test_encoded.drop(columns = ['tbp_lv_location_simple_Unknown','anatom_site_general_nan','tbp_lv_location','tbp_tile_type'], inplace=True)

df_test_encoded['age_approx'] = df_test_encoded['age_approx'].fillna(df_test_encoded['age_approx'].mean())

X_tab = df_encoded.drop(columns=['target'])
y = df_encoded['target']

smote = SMOTE(random_state=42)
X_tab_resampled, y_resampled = smote.fit_resample(X_tab, y)

target_samples = 15000  
classes, counts = np.unique(y_resampled, return_counts=True)
n_classes = len(classes)

from sklearn.utils import resample

samples_per_class = target_samples // n_classes

X_tab_reduced = []
y_reduced = []

for cls in classes:
    X_cls = X_tab_resampled[y_resampled == cls]
    y_cls = y_resampled[y_resampled == cls]
    
    X_cls_reduced, y_cls_reduced = resample(X_cls, y_cls,
                                            n_samples=samples_per_class,
                                            random_state=42)
    
    X_tab_reduced.append(X_cls_reduced)
    y_reduced.append(y_cls_reduced)

X_tab_reduced = pd.DataFrame(np.vstack(X_tab_reduced), columns=X_tab.columns)
y_reduced = np.hstack(y_reduced)
X_tab_reduced.head()

def load_and_preprocess_images(image_folder, df):
    image_data = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{int(row['id'])}.jpg")  
        img = Image.open(image_path).resize((128, 128))  
        img_array = np.array(img)
        if img_array.shape[-1] == 4:  
            img_array = img_array[:, :, :3]
        image_data.append(img_array)
    
    return np.array(image_data)

image_folder_path = os.path.join(project_dir, 'train_image')

X_images_processed = load_and_preprocess_images(image_folder_path, X_tab_reduced)

output_dir = 'processed_data'
os.makedirs(output_dir, exist_ok=True)

X_images_processed_file_path = os.path.join(output_dir, 'X_images_processed.npy')
X_tab_reduced_file_path = os.path.join(output_dir, 'X_tab_reduced.csv')
y_reduced_file_path = os.path.join(output_dir, 'y_reduced.npy')

np.save(X_images_processed_file_path, X_images_processed)
X_tab_reduced.to_csv(X_tab_reduced_file_path, index=False)
np.save(y_reduced_file_path, y_reduced)

print("Processed data has been saved successfully.")

output_dir = 'processed_data'
X_images_processed = np.load(os.path.join(output_dir, 'X_images_processed.npy'))
X_tab_reduced = pd.read_csv(os.path.join(output_dir, 'X_tab_reduced.csv'))
y_reduced = np.load(os.path.join(output_dir, 'y_reduced.npy'))


xgb_model = XGBClassifier(use_label_encoder=False)
cat_model = CatBoostClassifier(verbose=0)
lgbm_model = LGBMClassifier()

voting_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('cat', cat_model),
    ('lgbm', lgbm_model)],
    voting='soft'
)

voting_model.fit(X_tab_resampled, y_resampled)

# Define CNN model for image data
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    return model

cnn_model = create_cnn_model((128, 128, 3))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess images for training
def load_and_preprocess_images(image_folder, df):
    image_data = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_folder, f"{int(row['id'])}.jpg")  
        img = Image.open(image_path).resize((128, 128))  
        img_array = np.array(img)
        if img_array.shape[-1] == 4:  
            img_array = img_array[:, :, :3]
        image_data.append(img_array)
    
    return np.array(image_data)

image_folder_path = os.path.join(project_dir, 'train_image')
X_images_processed = load_and_preprocess_images(image_folder_path, df_encoded)

# Train CNN model on images
X_img_train, X_img_val, y_train, y_val = train_test_split(X_images_processed, y_resampled, test_size=0.2, random_state=42)
cnn_model.fit(X_img_train, y_train, epochs=10, validation_data=(X_img_val, y_val))

# Load test images for prediction
test_image_folder_path = os.path.join(project_dir, 'test_image')
X_test_images = load_and_preprocess_images(test_image_folder_path, df_test_encoded)

# Create a fusion model to combine CNN and voting model predictions
class FusionModel(models.Model):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.cnn_model = cnn_model
        self.voting_model = voting_model

    def call(self, inputs):
        image_input = inputs[0]
        tabular_input = inputs[1]
        
        cnn_output = self.cnn_model.predict(image_input)
        voting_output = self.voting_model.predict_proba(tabular_input)[:, 1]
        
        combined_output = (cnn_output.flatten() + voting_output) / 2  # Average predictions
        return np.where(combined_output > 0.5, 1, 0)  # Convert to binary predictions

fusion_model = FusionModel()

# Make predictions on the test set using the fusion model
test_predictions_binary = fusion_model([X_test_images, df_test_encoded])

# Create submission DataFrame
submission = pd.DataFrame({
    'id': df_test_encoded['id'],
    'target': test_predictions_binary.flatten()
})

# Save to CSV file
submission.to_csv('submission.csv', index=False)
print("Submission file has been created successfully.")



