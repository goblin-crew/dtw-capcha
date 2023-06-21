import stow
import tensorflow
from mltu.callbacks import TrainLogger, Model2onnx
import cv2
import tensorflow as tf
from mltu.losses import CTCloss
from mltu.metrics import CWERMetric
from mltu.model_utils import residual_block
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from configs import ModelConfigs
from mltu.dataProvider import DataProvider
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
from mltu.preprocessors import ImageReader


class DataPreprocessor:
    def __init__(self, method: int = cv2.IMREAD_COLOR, *args, **kwargs):
        self._method = method

    def __call__(self, image_path: str, label: str):
        return cv2.imread(image_path, self._method), label


def train_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    print("training model")
    inputs = tf.keras.layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x8 = residual_block(x7, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x9 = residual_block(x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = tf.keras.layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = tf.keras.layers.Dropout(dropout)(blstm)

    output = tf.keras.layers.Dense(output_dim + 1, activation='softmax', name="output")(blstm)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


# Take input parameters from launch if exists
optional_base_path = ''
# Load dataset
print("loading dataset")
dataset, vocab, max_len = [], set(), 0
for line in open("samples/index.txt"):
    parts = line.split(" ")
    dataset.append([stow.relpath(optional_base_path + parts[1]), parts[0]])
    vocab.update(list(parts[0]))
    max_len = max(max_len, len(parts[0]))

print("dataset loaded")
print("Generating configs")
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = sorted(list(vocab))
configs.max_text_length = max_len
configs.save()
print("configs saved")

# Define data provider
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader()],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
)
print("data provider defined")

train_data_provider, val_data_provider = data_provider.split(split=0.9)

train_data_provider.augmentors = [RandomBrightness(), RandomRotate(), RandomErodeDilate()]

# Define model
print("defining model")
model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

# Compile model
print("compiling model")
model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric()],
)

# Define callbacks
earlystopper = EarlyStopping(monitor='val_CER', patience=40, verbose=1)
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True,
                             mode='min')
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode='auto')
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
print("Starting to train model")
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)
print("model trained")

print("saving model")
# Save training and validation datasets as csv files
train_data_provider.to_csv(stow.join(configs.model_path, 'train.csv'))
val_data_provider.to_csv(stow.join(configs.model_path, 'val.csv'))
