from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
import numpy as np

validation = ImageDataGenerator(rescale=1 / 255)
validation_dataset = validation.flow_from_directory('basedata/validation/',
                                                    target_size=(200, 200),
                                                    batch_size=3,
                                                    class_mode='binary'
                                                    )
key_list = list(validation_dataset.class_indices.keys())
val_list = list(validation_dataset.class_indices.values())

model = models.load_model('cars.model')
img = image.load_img('Aston Martin_DB11_2017_211_20_600_52_12_76_50_186_15_RWD_4_2_2dr_bIv.jpg', target_size=(200, 200))

img = np.expand_dims(img, axis=0)
result = model.predict(np.vstack([img]))
position = val_list.index(result)
print(key_list[position])

