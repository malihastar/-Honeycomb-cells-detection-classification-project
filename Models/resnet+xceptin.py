# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import  Dense,Dropout,Flatten,Input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.models import  load_model
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns




# Data preparation
batch_size = 64
image_size=[224,224]
train_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/train'
test_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/test'



train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rescale=1./255,zoom_range=0.05,rotation_range=360,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.05)
test_datagen = ImageDataGenerator(rescale=1./255)
 #Add data generators for parsing images to the network while training
traning_set=train_datagen.flow_from_directory('C:/Users/MATLAB/Desktop/maliha_data/all_data/train',
                                            target_size=[224,224],
                                            batch_size=batch_size,
                                            class_mode='categorical',shuffle=True)
test_set=test_datagen.flow_from_directory('C:/Users/MATLAB/Desktop/maliha_data/all_data/test',
                                            target_size=[224,224],
                                            batch_size=batch_size,
                                            class_mode='categorical',shuffle=False)


full_name='concatenate'
classes_number=7 #Number of classes
input_tensor=Input(shape=(224,224,3))
######################################################################################################
base_model1 = Xception(weights='imagenet', include_top=False,input_tensor=input_tensor)
features1 = base_model1.output
######################################################################################################
base_model2 = ResNet50V2(weights='imagenet', include_top=False,input_tensor=input_tensor)
features2 = base_model2.output
concatenated=tf.keras.layers.concatenate([features1,features2]) #Concatenate the extracted features
####################################################################################################
conv=tf.keras.layers.Conv2D(1024, (1, 1),padding='same')(concatenated) #add the concatenated features to a convolutional layer
feature = Flatten(name='flatten')(conv)
dp = Dropout(0.5)(feature) #add dropout
preds = Dense(classes_number, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp) 
Concatenated_model = Model(inputs=input_tensor, outputs=preds)
#######################################################
for layer in Concatenated_model.layers:
  layer.trainable = True
Concatenated_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6), loss='categorical_crossentropy',metrics=['accuracy'])

  

H=Concatenated_model.fit(traning_set, epochs=30,validation_data=test_set,shuffle=True)
Concatenated_model.save("resnet1_xcepi_model.h5")


model = load_model("resnet1_xcepi_model.h5")

#Confusion Matrix and Classification Report
          
y_pred =model.predict(test_set,verbose=0,batch_size=batch_size)


predicted_proba = model.predict(test_set)
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

# train accuracy and val_accuracy graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('accuracy.png')
plt.show()
    
# train loss and val_loss graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('loss.png')
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
    plt.savefig('roc.png')
    plt.show()

plot_roc_curve(Concatenated_model,test_set, "resnet1_xcepi_model")

      