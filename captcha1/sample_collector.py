import random
import re

from PIL import Image, ImageDraw, ImageFont
import os


# Function to sanitize the filename
def sanitize_filename(filename):
    return re.sub(r"[^\w]", "", filename)


# Function to generate images of text
def generate_text_images(sequence, font_path, output_folder, file_nr):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Set the desired image size and font size
    image_size = (450, 40)
    font_size = 17

    # Load the TrueType font
    font = ImageFont.truetype(font_path, font_size)

    # Create a new blank image with a white background
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Calculate the text bounding box to center it in the image
    text_bbox = draw.textbbox((0, 0, image_size[0], image_size[1]), sequence, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    # Draw the text on the image
    draw.text((x, y), sequence, font=font, fill="black")

    # Sanitize the filename
    sanitized_sequence = sanitize_filename(sequence)

    # Save the image as a PNG file
    output_path = os.path.join(output_folder, f"{str(file_nr)}.png")
    image.save(output_path)
    # Write to file sequence and filename
    with open("samples/capcha-1/index.txt", "a") as f:
        f.write(f"{sequence} {output_path}\n")
    print(f"Saved image: {output_path}"
          f" ({remaining_samples}/{sample_count}) - {estimated_time:.2f}s left", end="\r")


# Recalculate estimated time, based on remaining samples and average download time
def recalculate_estimated_time(remaining_samples):
    global estimated_time
    estimated_time = remaining_samples * 0.5


def generateSequence(charset, length):
    sequence = ""
    for i in range(length):
        sequence += random.choice(charset)
    return sequence


sample_count = 1000
offset = 0
sample_output = "samples"
font_path = "unispace.ttf"
remaining_samples = sample_count
estimated_time = remaining_samples * 0.5
original_estimated_time = estimated_time
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@#$%&*?!+-"
sequence_length = 40

# Download samples
for i in range(offset, sample_count):
    recalculate_estimated_time(remaining_samples)
    generate_text_images(generateSequence(charset, sequence_length), font_path, sample_output, file_nr=i)
    remaining_samples -= 1
