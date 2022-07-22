
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix,classification_report
import itertools
from tensorflow.keras.models import  load_model
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd

# Data preparation
image_size=[80,80]
train_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/train'
test_path='C:/Users/MATLAB/Desktop/maliha_data/all_data/test'

train_datgen=ImageDataGenerator(rescale=1./155,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=0.2)

traing_set=train_datgen.flow_from_directory(train_path,
                                            target_size=[80,80],
                                            batch_size=128,
                                            class_mode='categorical')
test_set=train_datgen.flow_from_directory(test_path,
                                            target_size=[80,80],
                                            batch_size=128,
                                            class_mode='categorical',shuffle=False)



# Vgg16_model
vgg=VGG16(input_shape=image_size+[3],weights='imagenet',include_top=False)

for layer in vgg.layers:
  layer.trainable=False
classe=7
x=Flatten()(vgg.output)
prediction=Dense(classe,activation='softmax')(x)

model=Model(inputs=vgg.input,outputs=prediction)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

H=model.fit(traing_set,validation_data=test_set,epochs=30,steps_per_epoch=len(traing_set),validation_steps=len(test_set))

model.save('VGG16.h5')

model = load_model("VGG16.h5")

#Confusion Matrix and Classification Report
          
y_pred =model.predict(test_set,verbose=0,batch_size=16)


predicted_proba =model.predict(test_set)
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

#train accuracy and val_accuracy graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('accuracy.png')
plt.show()

    
#train loss and val_loss graphics

plt.figure(figsize=(7,7))

plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('loss.png')
plt.show()

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
    plt.savefig('Roc.png')
    plt.show()
    

plot_roc_curve(model,test_set, "VGG16")

      
