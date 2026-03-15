# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 16:39:23 2025

@author: bikra
"""

import imageio
import os

# === CONFIGURATION ===
folder_path = "C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/Animation/2024/Drawing_DA"  # Change to your folder path
output_gif = "C:/Users/bikra/Desktop/02. Bias_Forecast/Forecast/Animation/2024/animation_DA.gif"            # Output file name
duration =  800                     # Time per frame in seconds

# === READ IMAGES ===
images = []
# Sort so frames appear in order
file_list = sorted(os.listdir(folder_path))

for file_name in file_list:
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, file_name)
        images.append(imageio.imread(img_path))

# === CREATE GIF ===
imageio.mimsave(output_gif, images, duration=duration, loop=0)  # loop=0 → infinite loop

print(f"GIF saved as {output_gif}")
