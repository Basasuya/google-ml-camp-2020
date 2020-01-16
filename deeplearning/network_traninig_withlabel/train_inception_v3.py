from tensorflow.keras import layers
from tensorflow.keras import Model
import os

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

pre_trained_model = InceptionV3(input_shape = (224, 224, 3), 
                                include_top = False, 
                                weights='imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = True
  
# pre_trained_model.summary()

#last_layer = pre_trained_model.get_layer('mixed7')
#print('last layer output shape: ', last_layer.output_shape)
last_output = pre_trained_model.output

x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
#x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (37, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 


model.compile(optimizer='rmsprop', 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

bs = 12

# Define our example directories and files
base_dir = '/home/cyhong021/pet'

train_dir = os.path.join( base_dir, 'train')
#validation_dir = os.path.join( base_dir, 'validation')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

# Note that the validation data should not be augmented!
#test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = bs,
                                                    class_mode = 'categorical', 
                                                    target_size = (224, 224))     
from tensorflow.keras.callbacks import ModelCheckpoint
path = '/home/cyhong021/saved_model/inception_v3/'
filepath="trainable_model_epoch{epoch:02d}_loss{loss:.2f}_acc{acc:.2f}.h5"
checkpoint = ModelCheckpoint(path+filepath, monitor='loss', verbose=1, save_best_only=True, period=10)
history = model.fit_generator(
            train_generator,
            #validation_data = validation_generator,
            steps_per_epoch = 5911 // bs,
            epochs = 100,
           # validation_steps = 50,
            callbacks = [checkpoint],
            verbose = 1)