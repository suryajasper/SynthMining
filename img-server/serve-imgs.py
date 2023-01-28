from flask import Flask, jsonify

app = Flask(__name__)

PORT = 2003

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(port=PORT)