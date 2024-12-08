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
#Carga de librerias
from PIL import Image
import os
import numpy as np
#import cv2
# Ruta a la carpeta que contiene las imágenes
folder_path = 'Dataset\Todo'

# Lista para almacenar las imágenes cargadas
image_list = []
#Lista para almacenar las etiquetas
y = []
# Recorre todos los archivos en la carpeta
val = -1
for filename in os.listdir(folder_path):
    if 'Auto' in filename:
      y.append(0) #indice 0 Auto [Auto,Camion,Moto,Cuatrimoto,Camioneta]
      val = 0
    elif 'Camion' in filename: #indice 1 Camion
      y.append(1)
      val = 1
    elif 'Moto' in filename: #indice 2 Moto
      y.append(2)
      val = 2
    elif 'Cuatrimoto' in filename: #indice 3 Cuatrimoto
      y.append(3)
      val = 3
    elif 'Neta' in filename: #indice 4 Camioneta
      y.append(4)
      val = 4

    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        img_path = os.path.join(folder_path, filename)
        #img = Image.open(img_path)
        img = img.resize((224, 224))  # Redimensiona la imagen


        img_array = np.array(img) / 255.0  # Normaliza los valores de píxeles

        image_list.append(img_array)
        aumentadas = aumentarImagenes(img_array,2,True)
        image_list.extend(aumentadas)
        for i in range(len(aumentadas)):
          y.append(val)

#y2=y
#y2.sort()
#print(len(y2))
#Ordenamos la lista, ademas de mostrar la cantidad total y cuantos de cada etiqueta hay
#dict(zip(y2,map(lambda x: y2.count(x),y2)))

# Ahora 'image_list_train' contiene todas las imágenes cargadas