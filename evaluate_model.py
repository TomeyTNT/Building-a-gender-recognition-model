import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import itertools
import os
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('C:/Users/ahmadi/Desktop/New folder (2)/my_mode1_with_balance_earlystopping_15epoch.h5')

# Show the model architecture
new_model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/ahmadi/yolo_labeled_resized_split/train',
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'C:/Users/ahmadi/yolo_labeled_resized_split/val',
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/ahmadi/yolo_labeled_resized_split/test',
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary'
)

# # Get the ground truth labels
# true_labels = test_generator.classes

# # Get the predicted labels
# preds = new_model.predict(test_generator, steps=test_generator.samples)
# pred_labels = np.where(preds > 0.5, 1, 0)

# # Compute the confusion matrix
# cm = confusion_matrix(true_labels, pred_labels)
# print(cm)

# # Plot the confusion matrix
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], normalize=True, title='Normalized confusion matrix')

# plt.show()

# # Print classification report
# print(classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1']))

# train_loss, train_accuracy = new_model.evaluate(train_generator)
# print(f'Train accuracy: {train_accuracy}')

# test_loss, test_accuracy = new_model.evaluate(test_generator)
# print(f'Test accuracy: {test_accuracy}')

# اجرای پیش‌بینی بر روی تصاویر تست
predictions = new_model.predict(test_generator)

# تبدیل احتمالات به لیبل‌های نهایی
predicted_labels = (predictions > 0.5).astype(int)

print("Predicted Labels:", predicted_labels)


# ارزیابی ماژول Noise Classifier
true_labels = test_generator.labels

# محاسبه ماتریس درهم‌ریختگی
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# گزارش کامل ارزیابی
report = classification_report(true_labels, predicted_labels)
print("Classification Report:")
print(report)

defec = 0
okok=0

# مسیر ذخیره‌سازی نتایج پیش‌بینی
output_directory = 'C:/Users/ahmadi/Desktop/New folder'
for i in range(len(predicted_labels)):

    # مسیر فایل تصویر
    img_path = os.path.join(test_generator.directory, test_generator.filenames[i])
    img = image.load_img(img_path, target_size=(150, 150))
     # نام فایل
    file_name = os.path.basename(img_path)


    if(predicted_labels[i] == 0):
        class_output_directory = os.path.join(output_directory, 'Men')
        defec+=1
        output_path = os.path.join(class_output_directory, file_name)

        # ذخیره تصویر در مسیر مربوطه
        img.save(output_path)

        print(f'Image {i+1}/{len(predicted_labels)} saved at: {output_path}')
    elif(predicted_labels[i] == 1):
        class_output_directory = os.path.join(output_directory, 'Women')
        okok+=1
        output_path = os.path.join(class_output_directory, file_name)

        # ذخیره تصویر در مسیر مربوطه
        img.save(output_path)

        print(f'Image {i+1}/{len(predicted_labels)} saved at: {output_path}')

print("Defected: ", defec)
print("OK: ", okok)