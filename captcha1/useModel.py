import os

import cv2
import keras
import numpy as np
from keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, trainable, dtype, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # On test time, just return the computed loss
        return loss


def decode_batch_predictions(pred, characters, labels_to_char):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >= 0:
                outstr += labels_to_char[c]
        output_text.append(outstr)

    # return final text results
    return output_text


class UseModel:
    # Define image dimensions
    img_width = 450
    img_height = 40

    # Store all the characters in a set
    characters = set()

    # A list to store the length of each captcha
    captcha_length = []

    char_to_labels = {}
    labels_to_char = {}

    def __init__(self):
        model = keras.models.load_model('capcha-1.keras', custom_objects={"CTCLayer": CTCLayer})

        self.prediction_model = keras.models.Model(model.get_layer(name='input_data').input,
                                                   model.get_layer(name='dense2').output)
        self.load_data()

    def load_data(self):
        # Iterate over the dataset and store the
        # information needed
        for line in open("samples/index.txt"):
            parts = line.split(" ")
            # 1. Get the label associated with each image
            img_path = str.strip(parts[1]).replace('\\', '/')
            label = str.strip(parts[0])
            # 2. Store the length of this cpatcha
            self.captcha_length.append(len(label))

            # 4. Store the characters present
            for ch in label:
                self.characters.add(ch)

        # Sort the characters
        self.characters = sorted(self.characters)

        # Map text to numeric labels
        self.char_to_labels = {char: idx for idx, char in enumerate(self.characters)}

        # Map numeric labels to text
        self.labels_to_char = {val: key for key, val in self.char_to_labels.items()}

    def predict(self, img_path):
        img_path = str.strip(img_path).replace('\\', '/')
        img_path = img_path.replace('input/', '')
        img_path = os.path.join("input/", img_path)

        # Load and preprocess a single image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        # Apply thresholding to create a binary image
        _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        img = cv2.resize(img, (self.img_width, self.img_height))  # Resize to the desired dimensions

        img = img.astype(np.float32) / 255.0  # Normalize pixel values

        # Reshape the image to match the model's input shape
        img = img.reshape((1, self.img_height, self.img_width, 1)).T

        # Make predictions using the model
        prediction_text = self.prediction_model.predict(img)
        prediction_text = decode_batch_predictions(prediction_text, self.characters, self.labels_to_char)[0]
        print(prediction_text)
        return str(prediction_text)
