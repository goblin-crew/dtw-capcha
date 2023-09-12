import random
import re

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

from configs import ModelConfigs

configs = ModelConfigs()


# Function to sanitize the filename
def sanitize_filename(filename):
    return re.sub(r"[^\w]", "", filename)


# Function to generate images of text
def generate_image(char, rotate=True, **kwargs):
    # file_nr = str(char_nr) + str(int_nr + offset)
    # # Create the output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)

    # Set the desired image size and font size
    if 'image_size' in kwargs:
        image_size = kwargs['image_size']
    else:
        image_size = (configs.char_width, configs.char_height)

    char_img = Image.open(char['file'])
    new_img = char_img.resize(image_size)

    if rotate:
        rotation_fraction = (360 / 8) + random.randint(-5, 5)
        new_img = new_img.rotate(rotation_fraction * random.randint(0, 8), fillcolor=255, resample=Image.BILINEAR)

    new_img = new_img.convert("RGBA")
    # Blur the image
    # radius = random.randint(0, 2)
    # new_img = new_img.filter(ImageFilter.GaussianBlur(radius))

    return new_img

    # # save the image
    # new_img.save(os.path.join(output_folder, f"{str(file_nr)}.png"))
    #
    # # Save the image as a PNG file
    # output_path = f"{str(file_nr)}.png"
    #
    # # Write to file sequence and filename
    # with open(os.path.join(output_folder, "index.txt"), "a") as f:
    #     f.write(f"{char['char']} {output_path}\n")
    # print(f"Saved image: {output_path}"
    #       f" ({remaining_samples}/{sample_count}) - {estimated_time:.2f}s left", end="\r")


def generate_image_sequence(output_folder, file_nr):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    sequence = generateSequence(charMap, sequence_length)

    char_spacing = 5

    char_width = configs.width / sequence_length - char_spacing
    char_height = char_width

    img = Image.new('RGB', (configs.width, configs.height), color=(255, 255, 255))
    for i, char in enumerate(sequence):
        char_img = generate_image(char, image_size=(int(char_width), int(char_height)), rotate=False)
        vertival_offset = random.randint(-8, 8)
        horizontal_offset = random.randint(-4, 4)
        # img.paste(char_img, (int(i * char_width + (i + 1) * char_spacing), int(char_height / 2)))
        img.paste(char_img, (int(i * char_width + (i + 1) * char_spacing + horizontal_offset),
                             int(char_height / 2 + vertival_offset)), char_img)
    img.save(os.path.join(output_folder, f"{str(file_nr)}.png"))

    # Save the image as a PNG file
    output_path = f"{str(file_nr)}.png"

    char_sequence = "".join([char['char'] for char in sequence])

    # Write to file sequence and filename
    with open(os.path.join(output_folder, "index.txt"), "a") as f:
        f.write(f"{char_sequence} {output_path}\n")

    print(f"Saved image: {output_path}"
          f" ({remaining_samples}/{sample_count}) - {estimated_time:.2f}s left", end="\r")


# Recalculate estimated time, based on remaining samples and average download time
def recalculate_estimated_time(remaining_samples):
    global estimated_time
    estimated_time = remaining_samples * 0.5


def generateSequence(charset, length):
    sequence = []
    for i in range(length):
        sequence.append(charset[random.randint(0, len(charset) - 1)])
    return sequence


sample_count = 5000
offset = 0
sample_output = configs.data_dir

remaining_samples = sample_count
estimated_time = remaining_samples * 0.5
original_estimated_time = estimated_time

charMap = [
    {'symbol': ':D', 'char': 'A', 'file': 'emojis/0.png'},
    {'symbol': ':)', 'char': 'B', 'file': 'emojis/1.png'},
    {'symbol': ':P', 'char': 'C', 'file': 'emojis/2.png'},
    {'symbol': ':(', 'char': 'D', 'file': 'emojis/3.png'},
    {'symbol': ';)', 'char': 'E', 'file': 'emojis/4.png'},
    {'symbol': 'B)', 'char': 'F', 'file': 'emojis/5.png'},
    {'symbol': ':@', 'char': 'G', 'file': 'emojis/6.png'},
    {'symbol': ':o', 'char': 'H', 'file': 'emojis/7.png'},
    {'symbol': ':S', 'char': 'I', 'file': 'emojis/8.png'},
    {'symbol': ':|', 'char': 'J', 'file': 'emojis/9.png'},
    {'symbol': ':/', 'char': 'K', 'file': 'emojis/10.png'},
    {'symbol': '<3', 'char': 'L', 'file': 'emojis/11.png'}
]
# I only want the char
charset = [char['char'] for char in charMap]
sequence_length = configs.max_text_length

generate_image_sequence(sample_output, 0)

if offset <= 0:
    with open(os.path.join(sample_output, "index.txt"), "w") as f:
        f.write("")

for i in range(sample_count):
    generate_image_sequence(sample_output, i + 1)
    remaining_samples -= 1
    recalculate_estimated_time(remaining_samples)
