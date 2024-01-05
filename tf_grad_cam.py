import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
# 定义图像处理器
gen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.05,
    rotation_range=25
)


def plot_images(img):
    plt.figure(figsize=[12, 18])
    print(img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def make_gradcam_heatmap(image, model, last_conv_layer_name):
    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    # img_array = tf.expand_dims(img_array, axis=0)
    img_array=image
    # Remove last layer's softmax
    last_layer_activation = model.layers[-1].activation
    model.layers[-1].activation = None

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array[0].shape[1], img_array[0].shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.8 + img_array[0] * 255 * 0.8

    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Restore last layer activation
    model.layers[-1].activation = last_layer_activation

    return superimposed_img

# 读取单张图像并进行处理
img_path = 'D:/Desktop/test/test/分析.png'  # 替换为你的图像路径
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(300, 300))  # 加载图像并调整尺寸
img_array =tf.keras.preprocessing.image.img_to_array(img)  # 将图像转换为数组
# img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度，因为 flow() 接受批量数据
img_array = tf.expand_dims(tf.convert_to_tensor(img_array), axis=0)

# 使用图像处理器进行处理
gen.fit(img_array)

# 使用 flow() 方法获取处理后的图像数据
processed_img = next(gen.flow(img_array))
vgg16_model = load_model('D:/Desktop/test/test/model')

vgg16_test_preds = vgg16_model.predict(processed_img)
vgg16_test_pred_classes = np.argmax(vgg16_test_preds, axis=1)

last_conv_layer_name = "block5_conv3"



heatmaps = make_gradcam_heatmap(processed_img, vgg16_model, last_conv_layer_name)

plot_images(heatmaps)
