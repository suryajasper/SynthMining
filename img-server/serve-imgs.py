from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from os import path, listdir, remove, rename, mkdir
import json
import base64
import requests

import cv2
from PIL import Image

app = Flask(__name__)

PORT = 2003

@app.route('/')
def hello():
  return 'Hello, World!'

@app.route('/uploadImages', methods=['POST'])
@cross_origin()
def upload_images():
  batch_size = int(request.form.get('batch_size'))
  project_id = request.form.get('project_id')

  dir_path = f'imgs/{project_id}'

  fs = [request.files.get(str(i)) for i in range(batch_size)]

  if not path.exists(dir_path):
    mkdir(dir_path)

  for file_obj in fs:
    file_obj.save(path.join(dir_path, file_obj.filename))

  filelist = [ f for f in listdir('temp') if path.isfile(path.join('temp', f)) ]

  for f in filelist:
    remove(path.join('temp', f))

  return batch_size

@app.route('/getProjectImages', methods=['GET'])
@cross_origin()
def get_images():
  project_id = request.args.get('project_id', type=str)
  max_imgs = request.args.get('max_imgs', default=30, type=int)
  category = request.args.get('category', default=None, type=str)

  dir_path = path.join('imgs', project_id)
  if not path.exists(dir_path):
    print(dir_path, 'does not exist')
    return {'images': []}

  images = []
  # stored = listdir(dir_path)

  attr_lines = [line.strip() for line in open('formatted_attrs.txt').readlines()]

  i = 0

  while len(images) < max_imgs:
    spl = attr_lines[i+1].split(' ')
    fname = spl[0]
    categories = spl[1].split(',')

    print(i, category)
    if category is not None and category not in attr_lines[i+1]:
      i += 1
      continue 

    images.append({
      'src': to_base64(path.join(dir_path, fname)),
      'name': fname.split('.')[0],
      'categories': categories,
    })

    i += 1
  
  return {
    'images': images,
    'categories': attr_lines[0].split(' '),
  }


def to_base64(file_name):
  data = ''
  with open(file_name , "rb") as image_file :
    data = base64.b64encode(image_file.read())
  return data.decode('utf-8')

if __name__ == '__main__':
  app.run(port=PORT)