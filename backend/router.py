from flask import Flask, request
from src.Image import Image
import json
from werkzeug.wrappers import Response, Request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

global image
choose="vgg_label"
image = Image(data="./data/" + choose + "/data.json", embedding="./data/" + choose + "/embedding.npz")
image.constructKnn()

@app.route('/getTSNE')
def getTSNE():
    global image
    # graph = Graph('./data/all.json')
    result = image.getDimension()
    return Response(json.dumps(result), content_type="application/json")

@app.route('/getKNN', methods=['GET', 'POST'])
def getKNN():
    global image
    data = json.loads(request.get_data())
    # print(data)
    # print(data['choose'])
    result = image.getKnn(data['id'], data['k'])
    return Response(json.dumps(result), content_type="application/json")


if __name__ == '__main__':
    app.run(debug=True)
    # app.run()