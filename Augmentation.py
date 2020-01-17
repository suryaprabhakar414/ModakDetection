# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
import os
from tqdm import tqdm 
datagen = ImageDataGenerator( 
		rotation_range = 40, 
		shear_range = 0.2, 
		zoom_range = 0.2, 
		horizontal_flip = True, 
		brightness_range = (0.5, 1.5)) 
	
path = "D:/Python Scripts/Intern/test"

for imag in tqdm(os.listdir(path)):

    i = 0
    img = load_img(os.path.join(path,imag)) 
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)
    for batch in datagen.flow(x, batch_size = 1, save_to_dir ='D:/Python Scripts/Intern/New Folder/Non-Modak',save_prefix ='image', save_format ='jpeg'):
        i += 1
        if i > 7: 
            break
    
