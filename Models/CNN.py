# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:35:33 2021

@author: MATLAB
"""
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
from keras.models import  load_model
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
BATCH_SIZE=1
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

for d in devices:
    t = d.device_type
    name = d.physical_device_desc
    l = [item.split(':',1) for item in name.split(", ")]
    name_attr = dict([x for x in l if len(x)==2])
    dev = name_attr.get('name', 'Unnamed device')
    print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")

GPUS = ["GPU:0","GPU:1","GPU:2"]

BATCH_SIZE=64
image_size=[32,32]
train_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/train'
test_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/test'

strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync)

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rescale=1./255,zoom_range=0.05,rotation_range=360,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.05)
test_datagen = ImageDataGenerator(rescale=1./255)

train_datgen=ImageDataGenerator(rescale=1./155,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=0.2)

traing_set=train_datgen.flow_from_directory(train_path,
                                            target_size=[32,32],
                                            batch_size=batch_size,
                                            class_mode='categorical')
test_set=train_datgen.flow_from_directory(test_path,
                                            target_size=[32,32],
                                            batch_size=batch_size,
                                            class_mode='categorical',shuffle=False)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2),padding='same', activation='relu', input_shape=image_size+[3]))
model.add(Conv2D(filters=16, kernel_size=(2,2),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=32, kernel_size=(3,3),padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.3))
model.add(Conv2D(filters=64, padding='same',kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(filters=64,padding='same', kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.35))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(7, activation='softmax'))
#model.summary()


model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

H=model.fit(traing_set,epochs=30,validation_data=test_set,batch_size=batch_size,  verbose=1)

model.save('my_model.h5')



model = load_model("my_model.h5")

#Confusion Matrix and Classification Report
          
y_pred =model.predict(test_set,verbose=0,batch_size=batch_size)


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

#accuracy and val_accuracy graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('inception_ct_accuracy.png')
plt.show()
    
#loss and val_loss graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('inception_dem_loss.png')
plt.show()



fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()

n_class = 7

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(test_set.classes, y_pred[:,i], pos_label=i)

    roc_auc[i] = auc(fpr[i], tpr[i])



def plot_roc_curve(model, val,  model_name):
    predictions = model.predict(test_set, steps=1)
    
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

plot_roc_curve(model,test_set, "my_model")

      



 
