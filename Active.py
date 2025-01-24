import tensorflow as tf
import cv2
import os
import numpy as np
import imghdr
import matplotlib
from matplotlib import pyplot as plt
# Import augmentation layers
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization




Desk_path = os.path.expanduser('~/Desktop')
project_path = 'NN Mushroom'
data_path = 'dataset'
data_add = os.path.join(Desk_path, project_path, data_path)
viable_extensions = ['jpeg', 'jpg', 'png']






# Testing if it works
# print(os.path.exists(Desk_path))
# print(os.listdir(os.path.join(Desk_path, project_path, data_path, 'Edible')))

# #MAKING SURE ALL OF THE DATA IS VIABLE BY REMOVING UNWANTED EXTENSIONS AND UNREADBLE IMAGES
for subset in os.listdir(data_add):
    if subset != ".DS_Store":
        for img in os.listdir(os.path.join(data_add, subset)):
            try:
                img_path = os.path.join(data_add, subset, img)
                # Check if the image can be opened and is not corrupt
                with open(img_path, 'rb') as f:
                    is_valid = tf.io.decode_jpeg(f.read())
            except:
                print(f"Removing corrupt image: {img_path}")
                os.remove(img_path)
for subset in os.listdir(os.path.join(data_add)):
    if(subset!=".DS_Store"):
        for img in os.listdir(os.path.join(data_add, subset)):
            try:
                image = cv2.imread(os.path.join(data_add, subset, img))
                img_ext = imghdr.what(os.path.join(data_add, subset, img))
                if(img_ext not in viable_extensions):
                    print("not viable extension")
                    os.remove(os.path.join(data_add, subset, img))
            except:
                print("can't be read by cv2")
                os.remove(os.path.join(data_add, subset, img))
                


# command in keras that batches the data in groups of 32 and puts each thing in their respective array(images and labels)
#batch[0] contains images and batch[1] contains labels(edible or inedible)
data = tf.keras.utils.image_dataset_from_directory(data_add)

#data[0] cannot be performed on data, so iterator must be used
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#0 IS No, 1 Yes    

# LOOP TO DISPLAY THE PICTURES
#fig is the figure and ax is the array of axes. ncols is the number of pictures displayed, figsize is size of figures. 
#enumerate just makes it so idx increments from 0 to 4, img is the img itself
#imshow requires the img to be integers to work =and the img may be a float type, so we use .astype(int) to convert to integers
fig, ax = plt.subplots(ncols=5, figsize=(20,20))
for idx, img in enumerate(batch[0][:5]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
# plt.show()

#PREPROCESSING THE DATA
#SCALING

# Define augmentation pipeline
data_augmentation = Sequential([
    RandomFlip("horizontal"),        # Randomly flip images horizontally
    RandomRotation(0.2),             # Randomly rotate images by 20%
    RandomZoom(0.2),                 # Randomly zoom in/out
    RandomContrast(0.2)              # Randomly adjust contrast
])

# Apply augmentation and scaling
# Step 1: Apply augmentation and scaling
# Load dataset and split into training and validation sets using image_dataset_from_directory
data = tf.keras.utils.image_dataset_from_directory(data_add)

#data[0] cannot be performed on data, so iterator must be used
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#0 IS Absent, 1 Present    

# LOOP TO DISPLAY THE PICTURES
#fig is the figure and ax is the array of axes. ncols is the number of pictures displayed, figsize is size of figures. 
#enumerate just makes it so idx increments from 0 to 4, img is the img itself
#imshow requires the img to be integers to work =and the img may be a float type, so we use .astype(int) to convert to integers
fig, ax = plt.subplots(ncols=5, figsize=(20,20))
for idx, img in enumerate(batch[0][:5]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
# plt.show()

#PREPROCESSING THE DATA
#SCALING

# Define augmentation pipeline
data_augmentation = Sequential([
    RandomFlip("horizontal"),        
    RandomRotation(0.1),             
    RandomZoom(0.1),                 
    RandomContrast(0.1)              
])

# Apply augmentation and scaling
# Step 1: Apply augmentation and scaling
scaled_data = data.map(lambda x, y: (data_augmentation(x) / 255.0, y))

# Step 2: Shuffle the dataset (reshuffle each epoch for better generalization)
# scaled_data = scaled_data.shuffle(buffer_size=500, reshuffle_each_iteration=True)

# Step 3: Optional - Prefetch data for optimized loading
# scaled_data = scaled_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# Step 4: Iterate through the data to inspect batches (if needed)
scaled_iterator = scaled_data.as_numpy_iterator()
scaled_batch = scaled_iterator.next()

# Step 5: Optionally print or inspect the batch
# print(scaled_batch)



#loop to show scaled data, this time dont convert img to integers as that is only needed for unnormalized data
#if converted to integers here, then all values would be zero as they are all between 0 and 1
fig, ax = plt.subplots(ncols=5, figsize=(20,20))
for idx, img in enumerate(scaled_batch[0][:5]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(scaled_batch[1][idx])
# plt.show()


#SPLITTING DATA INTO TRAINING(0.6), CROSS VALIDATION(0.2), AND TESTIING SETS(0.2)
print("total: ")
print(len(scaled_data))
TRAIN_SIZE = int(0.7*len(scaled_data))+1
VAL_SIZE = int(0.2*len(scaled_data))+1 
TEST_SIZE = int(0.1*len(scaled_data))
print("\n Tr + Te + Va: ")
print(TRAIN_SIZE+VAL_SIZE+TEST_SIZE)
train = scaled_data.take(TRAIN_SIZE)
val = scaled_data.skip(TRAIN_SIZE).take(VAL_SIZE)
test = scaled_data.skip(TRAIN_SIZE+ VAL_SIZE).take(TEST_SIZE)
# print(len(train))

#BUILDING THE NETWORK
#We will be using convolutional neural networks as they work better with grid-like structures like pictures

# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(256, 256, 3), include_top=False, weights='imagenet')
# base_model.trainable = False  # Freeze the pre-trained layers

model = Sequential()
model.add(Conv2D(32, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
logdir = '/Users/unmeshreza/Desktop/NN Mushroom/Logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
earlystopper = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15)

# Define the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',        
    factor=0.5,                
    patience=7,                
    min_lr=1e-6,               
    verbose=1                  
)

# Add this callback to your model's fit method
hist = model.fit(
    train,
    epochs=100,
    validation_data=val,
    callbacks=[tensorboard_callback, earlystopper
            #    , reduce_lr
               ]
)

print(hist.history)

fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='loss')
plt.plot(hist.history['val_loss'], color='red', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result().numpy(), re.result().numpy(), acc.result().numpy())
save_dir = os.path.join('/Users/unmeshreza/Desktop/NN Mushroom/Model')
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, 'TumorModel.h5'))

