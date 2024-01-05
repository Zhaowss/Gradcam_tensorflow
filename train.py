import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from keras.utils import image_dataset_from_directory
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.models import Model


FINE_TUNING_EPOCHS = 3
TRAINING_EPOCHS = 1
BATCH_SIZE = 64

image_height = 300
image_width = 300


# 获取当前脚本所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))


base_dataset = image_dataset_from_directory(
    current_dir,
    image_size=(image_height, image_width),
    crop_to_aspect_ratio=True,
    shuffle=False,
    batch_size=32)

base_df = pd.DataFrame(base_dataset.file_paths.copy())
base_df.columns = ['fullpaths']
base_df['labels'] = base_df.apply(
    lambda x: base_dataset.class_names[1] if (base_dataset.class_names[1] in x.fullpaths) else base_dataset.class_names[
        0], axis=1)
base_df['filepaths'] = base_df.apply(lambda x: str(x.fullpaths).replace(current_dir, ''), axis=1)

pd.set_option('display.max_colwidth', None)
base_df.head(10)
freq = base_df['labels'].value_counts()
print(freq)

freq.plot(kind='pie', figsize=(5, 5), title='Surface Cracks', autopct='%1.1f%%', shadow=False, fontsize=8);

train_df, test_df = np.split(base_df.sample(frac=1, random_state=42), [int(.8 * len(base_df))])

gen = ImageDataGenerator(rescale=1. / 255.,
                         horizontal_flip=True,
                         vertical_flip=True,
                         zoom_range=0.05,
                         rotation_range=25)

train_generator = gen.flow_from_dataframe(
    train_df,  # dataframe
    directory=current_dir,  # images data path / folder in which images are there
    x_col='fullpaths',
    y_col='labels',
    color_mode="rgb",
    target_size=(image_height, image_width),  # image height , image width
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42)

# Data agumentation and pre-processing using tensorflow
test_gen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_gen.flow_from_dataframe(
    test_df,  # dataframe
    directory=current_dir,  # images data path / folder in which images are there
    x_col='fullpaths',
    y_col='labels',
    color_mode="rgb",
    target_size=(image_height, image_width),  # image height , image width
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False)

# Get labels in dataset
a = train_generator.class_indices
class_names = list(a.keys())  # storing class/breed names in a list


def create_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(len(class_names), activation='softmax')(x)

    model = Model(base_model.inputs, outputs)

    return model


def fit_model(model, base_model, epochs, fine_tune=0):
    early = tf.keras.callbacks.EarlyStopping(patience=10,
                                             min_delta=0.001,
                                             restore_best_weights=True)
    # early stopping call back

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.

    print("Unfreezing number of layers in base model = ", fine_tune)

    if fine_tune > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune]:
            layer.trainable = False
            # small learning rate for fine tuning
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        base_model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # categorical cross entropy is taken since its used as a loss function for
    # multi-class classification problems where there are two or more output labels.
    # using Adam optimizer for better performance
    # other optimizers such as sgd can also be used depending upon the model

    # fit model
    history = model.fit(train_generator,
                        epochs=epochs,
                        callbacks=[early])

    model.save('D:/Desktop/test/test/model')
    return history

vgg16_base_model = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(image_height, image_width, 3)
)

vgg16_model = create_model(vgg16_base_model)
history = fit_model(vgg16_model, vgg16_base_model, epochs=TRAINING_EPOCHS)
