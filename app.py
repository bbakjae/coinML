from flask import Flask, jsonify, request

# from flask_restx import Resource, Api

app = Flask(__name__)
# api = Api(app)


@app.route("/covidResult", methods=["POST"])
def hello(self):
    return {"just": "fun"}


if __name__ == "__main__":
    app.run(host="0.0.0.0")
