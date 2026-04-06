import os
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image


BASE_DIR = os.path.dirname(__file__)


TRAIN_PATH = os.path.join(BASE_DIR, "data",  'train')
TEST_PATH = os.path.join(BASE_DIR, "data",  'test')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH, image_size=(128, 128), batch_size=  20,
    label_mode="categorical"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH, image_size=(128, 128), batch_size=  20,
    label_mode="categorical"
)


model = models.Sequential()

model.add(layers.Rescaling(1./255, input_shape=(128, 128, 3)))
model.add(layers.Conv2D(32, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2))) #vизнаchає головне на картинці

model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())  #картинку в числa

model.add(layers.Dense(64, activation = "relu"))

model.add(layers.Dense(3, activation="softmax"))

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy", #ціна похибки
    metrics = ["categorical_accuracy"])


model.fit(train_ds, epochs = 20, validation_data=test_ds)

test_photo = os.path.join(BASE_DIR, "images" 'test.format')

model.add(layers.Dense(3, activation="softmax"))

if os.path.exists(test_photo):
    img = image.load_img(test_photo, target_size = (128, 128))
    img_array = image.img_to_array(img)



    predictions = model.predict(img_array)
    class_name = sorted(os.listdir(TRAIN_PATH))

    result_ind = numpy.argmax(predictions[0])

    print(f"Результат: {class_name[result_ind]}")