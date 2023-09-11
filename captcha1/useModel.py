import keras

model = keras.models.load_model('kaggle/capcha-1.keras')

prediction_model = keras.models.Model(model.get_layer(name='input_data').input,
                                      model.get_layer(name='dense2').output)
prediction_model.summary()


# # A utility to decode the output of the network
# def decode_batch_predictions(pred):
#     pred = pred[:, :-2]
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#
#     # Use greedy search. For complex tasks, you can use beam search
#     results = keras.backend.ctc_decode(pred,
#                                        input_length=input_len,
#                                        greedy=True)[0][0]
#
#     # Iterate over the results and get back the text
#     output_text = []
#     for res in results.numpy():
#         outstr = ''
#         for c in res:
#             if c < len(characters) and c >= 0:
#                 outstr += labels_to_char[c]
#         output_text.append(outstr)
#
#     # return final text results
#     return output_text


