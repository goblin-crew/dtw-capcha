import cv2
import keras
import numpy as np
from keras import layers

# Store all the characters in a set
characters = set()

# A list to store the length of each captcha
captcha_length = []

# Store image-label info
dataset = []

# Iterate over the dataset and store the
# information needed
for line in open("samples/index.txt"):
    parts = line.split(" ")
    # 1. Get the label associated with each image
    img_path = str.strip(parts[1]).replace('\\', '/')
    label = str.strip(parts[0])
    # 2. Store the length of this cpatcha
    captcha_length.append(len(label))

    # 4. Store the characters present
    for ch in label:
        characters.add(ch)

# Sort the characters
characters = sorted(characters)

# Map text to numeric labels
char_to_labels = {char: idx for idx, char in enumerate(characters)}

# Map numeric labels to text
labels_to_char = {val: key for key, val in char_to_labels.items()}


#
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


model = keras.models.load_model('capcha-1.keras', custom_objects={"CTCLayer": CTCLayer})

prediction_model = keras.models.Model(model.get_layer(name='input_data').input,
                                      model.get_layer(name='dense2').output)


# A utility to decode the output of the network
def decode_batch_predictions(pred):
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


# Define image dimensions
img_width = 450
img_height = 40

# Load and preprocess a single image
img = cv2.imread("samples/0.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
img = cv2.resize(img, (img_width, img_height))  # Resize to the desired dimensions
img = img.astype(np.float32) / 255.0  # Normalize pixel values

# Reshape the image to match the model's input shape
img = img.reshape((1, img_height, img_width, 1)).T

# Make predictions using the model
prediction_text = prediction_model.predict(img)
prediction_text = decode_batch_predictions(prediction_text)
print(prediction_text)
