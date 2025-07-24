from flask import Flask, request, jsonify
from methods import *


app = Flask(__name__)

@app.route('/')
def home():
	return "Hello world"



@app.route('/add_image', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('data\\'+f.filename)
        res= add_image_feature('data\\'+f.filename)
        print(res)
        return jsonify({'outcome':res})
    

@app.route('/take_attendance', methods = ['GET', 'POST'])
def take_attendance():
    
    if request.method == 'POST':
        f = request.files['file']
        f.save('attendance\\'+f.filename)
        res= match_fingerprint('attendance\\'+f.filename)
        return jsonify({'outcome':res})

if __name__ == "__main__":
    app.run(debug=True)