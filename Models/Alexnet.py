# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:10:38 2021

@author: MATLAB
"""
import tensorflow as tf
from tensorflow.keras.layers import  Dense,Dropout, MaxPooling2D,Flatten,Conv2D,Activation
from tensorflow.keras.layers.normalization import BatchNormalization 
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.models import  load_model
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

# Data preparation
train_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/train'
test_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/test'
train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rescale=1./255,zoom_range=0.05,rotation_range=360,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.05)
test_datagen = ImageDataGenerator(rescale=1./255)
traning_set=train_datagen.flow_from_directory(train_path,
                                            target_size=[227,227],
                                            batch_size=128,
                                            class_mode='categorical',shuffle=True)
test_set=test_datagen.flow_from_directory(test_path,
                                            target_size=[227,227],
                                            batch_size=128,
                                            class_mode='categorical',shuffle=False)


# AlextNet model
Alexnet=Sequential()
Alexnet.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=(4,4), padding='same'))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

Alexnet.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))


Alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))


Alexnet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))


Alexnet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

Alexnet.add(Flatten())
Alexnet.add(Dense(4096, input_shape=(227,227,3,)))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(Dropout(0.4))

Alexnet.add(Flatten())
Alexnet.add(Dense(4096))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('relu'))
Alexnet.add(Dropout(0.4))

Alexnet.add(Flatten())
Alexnet.add(Dense(7))
Alexnet.add(BatchNormalization())
Alexnet.add(Activation('softmax'))

Alexnet.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('\n Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

H=Alexnet.fit_generator(traning_set, epochs=30,validation_data=test_set,shuffle=True,callbacks=[EarlyStoppingAtMinLoss()])

Alexnet.save("Alexnet.h5")



model = load_model("Alexnet.h5")

# Confusion Matrix and Classification Report
          
y_pred =model.predict(test_set,verbose=0,batch_size=16)


predicted_proba = model.predict_proba(test_set)
roc_auc = roc_auc_score(test_set.classes, predicted_proba, multi_class='ovr')

print("AUC SCORE ",roc_auc)


y_pred_bin=np.argmax(y_pred,axis=1)


cm = confusion_matrix(test_set.classes, y_pred_bin)
target_names = ['0', '1', '2','3','4','5','6']
print(classification_report(test_set.classes, y_pred_bin, target_names=target_names))

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    print(cm)
  

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm_plot_labels = ['0','1','2','3','4','5','6']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# Train accuracy and val_accuracy graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('inception_ct_accuracy.png')
plt.show()
plt.savefig('a.png')
# Train loss and val_loss graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('inception_dem_loss.png')
plt.show()
plt.savefig('b.png')


fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()

n_class = 7

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(test_set.classes, y_pred[:,i], pos_label=i)

    roc_auc[i] = auc(fpr[i], tpr[i])



def plot_roc_curve(model, val,  model_name):
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_test_dummies = pd.get_dummies(test_set.classes, drop_first=False).values
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {model_name}')
    for i in range(7):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %i' % (roc_auc[i], i))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
    plt.savefig('C.png')
plot_roc_curve(model,test_set, "Alexnet")

      



 


