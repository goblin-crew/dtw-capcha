import cv2
import numpy as np
from matplotlib import pyplot as plt


def getContours(img_path='image.jpg'):
    # Define the lower and upper bounds for the color of emojis (adjust these as needed)
    lower_color = np.array(0)
    upper_color = np.array(200)

    # Load and preprocess a single image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    color_mask = cv2.inRange(img, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Define a minimum and maximum size for emojis (adjust these as needed)
    min_size = 80
    max_size = 1000

    detected_emojis = []

    # Sort contours from left to right
    # contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Sort contours from top to bottom
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])
    # Use OpenCV to detect and draw rectangles around emojis
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter contours based on size
        if min_size < area < max_size:
            x, y, w, h = cv2.boundingRect(contour)
            img_rect = cv2.resize(img[y - 2:y + h + 2, x - 2:x + w + 2], (30, 30))
            detected_emojis.append(img_rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save or display the image with rectangles
    if __name__ == '__main__':
        cv2.imwrite('output_image.png', img)
        cv2.imshow('Image with Rectangles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_emojis, contours


if __name__ == '__main__':
    detected_emojis, _ = getContours('image.jpg')
    # Save to emojis folder enumerated
    for i, emoji in enumerate(detected_emojis):
        cv2.imwrite('emojis/' + str(i) + '.png', emoji)
