import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Parameters ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# --- Dataset Paths ---
train_path = 'archive/Diagnosis of Diabetic Retinopathy/train/'
valid_path = 'archive/Diagnosis of Diabetic Retinopathy/valid/'

# --- Load and Combine Datasets ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)
valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

# --- Combine and Shuffle ---
full_ds = train_ds.concatenate(valid_ds).shuffle(buffer_size=1000, seed=42)

# --- Split into Train and Validation ---
total_batches = tf.data.experimental.cardinality(full_ds).numpy()
train_size = int(0.8 * total_batches)
train_ds_final = full_ds.take(train_size)
val_ds_final = full_ds.skip(train_size)

# --- Preprocess ---
def preprocess(image, label):
    return preprocess_input(image), label

train_ds_final = train_ds_final.map(preprocess).prefetch(AUTOTUNE)
val_ds_final = val_ds_final.map(preprocess).prefetch(AUTOTUNE)

# --- Build MobileNetV2 Model ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# --- Freeze Base Layers ---
for layer in base_model.layers:
    layer.trainable = False

# --- Compile ---
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# --- Train CNN ---
model.fit(train_ds_final, validation_data=val_ds_final, epochs=EPOCHS)

# --- Save Model ---
model.save("dr_detector_mobilenetv2.h5")
print("✅ CNN model trained and saved.")

# --- CNN Evaluation ---
# Gather predictions
y_true = []
y_pred = []

for images, labels in val_ds_final:
    preds = model.predict(images)
    y_pred.extend((preds > 0.5).astype(int).flatten())
    y_true.extend(labels.numpy())

# --- Classification Report ---
print("CNN Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=['DR', 'No_DR']))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DR', 'No_DR'], yticklabels=['DR', 'No_DR'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CNN Confusion Matrix')
plt.show()

# --- Feature Extraction for SVM ---
feature_extractor = Model(inputs=base_model.input, outputs=x)

def extract_features(dataset):
    features = []
    labels = []
    for images, lbls in dataset:
        f = feature_extractor.predict(images)
        features.append(f)
        labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)  # ✅ FIXED HERE

# Extract and normalize
train_features, train_labels = extract_features(train_ds_final)
val_features, val_labels = extract_features(val_ds_final)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# --- Train SVM ---
svm = SVC(kernel='rbf', probability=True)
svm.fit(train_features, train_labels)

# --- SVM Evaluation ---
svm_preds = svm.predict(val_features)
print("SVM Classification Report:\n")
print(classification_report(val_labels, svm_preds, target_names=['DR', 'No_DR']))

# --- SVM Confusion Matrix ---
cm_svm = confusion_matrix(val_labels, svm_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=['DR', 'No_DR'], yticklabels=['DR', 'No_DR'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

