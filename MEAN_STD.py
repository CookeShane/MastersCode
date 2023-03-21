import numpy as np
import cv2
import os

# # Load dataset of images
# img_dir = './Dataset/HRI/Control/set20/'
# images = []
# for image_name in os.listdir(img_dir):
#     img = cv2.imread(f'./Dataset/HRI/Control/set20/{image_name}')
#     images.append(img)

# # Convert images to numpy array format
# images_array = np.array(images)

# # Compute mean and standard deviation for each channel
# mean = np.mean(images_array, axis=(0,1,2))/255
# std = np.std(images_array, axis=(0,1,2))/255

# print('Mean:', mean)
# print('Standard deviation:', std)

mean1 = [[0.40054006, 0.42260391, 0.39076512], [0.48170074, 0.51185119, 0.49546295], [0.44247466, 0.4368605,  0.42809254], [0.46578774, 0.46119042, 0.4326607 ], [0.51205847, 0.51817129, 0.48937604], [0.45182708, 0.45002719, 0.43585468], [0.58557782, 0.56206702, 0.55087716]]
mean = np.mean(mean1, axis=(0,1,2))/255
print('Mean:', mean)