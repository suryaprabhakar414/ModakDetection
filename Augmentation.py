from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
import os
from tqdm import tqdm 
datagen = ImageDataGenerator( 
		rotation_range = 40, 
		shear_range = 0.2, 
		zoom_range = 0.2, 
		horizontal_flip = True, 
		brightness_range = (0.5, 1.5)) 
	
path_modak = "D:/Python Scripts/Project/Modak"

for imag in tqdm(os.listdir(path_modak)):
    i = 0
    img = load_img(os.path.join(path_modak,imag)) 
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)
    for batch in datagen.flow(x, batch_size = 1, save_to_dir ='D:/Python Scripts/Project/Train/Modak',save_prefix ='image', save_format ='jpeg'):
        i += 1
        if i > 6: 
            break
    
path_non_modak = "D:/Python Scripts/Project/Non_Modak"

for imag in tqdm(os.listdir(path_non_modak)):
    i = 0
    img = load_img(os.path.join(path_non_modak,imag)) 
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)
    for batch in datagen.flow(x, batch_size = 1, save_to_dir ='D:/Python Scripts/Project/Train/Non-Modak',save_prefix ='image', save_format ='jpeg'):
        i += 1
        if i > 6: 
            break
