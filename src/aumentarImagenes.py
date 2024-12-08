from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
def aumentarImagenes(img_array,n,plotear=False):
  img_array = np.expand_dims(img_array,axis=0)

  datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
  )
  i = 0
  images = []
  for batch in datagen.flow(img_array, batch_size=1):
    if plotear:
      plt.figure(i)
      plt.imshow(batch[0])
    images.append(batch[0])
    i += 1
    if i >= n:
        break
  if plotear:
    plt.show()
  return images