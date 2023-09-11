from flask import Flask, request
from flask_cors import CORS

from captcha1.useModel import UseModel

app = Flask(__name__)
CORS(app)

ocr = UseModel()


@app.route("/", methods=['POST'])
def get_captcha():
    f = request.files['image']
    f.save('input/captcha.png')
    result = ocr.predict('captcha.png')
    return {"result": result}


