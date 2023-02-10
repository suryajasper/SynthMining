from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from os import path, listdir, remove, rename, mkdir
import json
import requests

import img_transforms

app = Flask(__name__)

PORT = 2003

DIR_PATH = 'imgs'

@app.route('/uploadImages', methods=['POST'])
@cross_origin()
def upload_images():
  MAX_IMG_SIZE = 300
  
  batch_size = request.form.get('batch_size', type=int)
  img_ids = request.form.getlist('img_ids[]')

  print(batch_size, img_ids)

  if not batch_size or not img_ids or batch_size != len(img_ids):
    return 'bad request', 400

  fs = [request.files.get(str(i)) for i in range(batch_size)]

  if not path.exists(DIR_PATH):
    mkdir(DIR_PATH)

  for i, file_obj in enumerate(fs):
    file_ext = file_obj.filename.split('.')[-1].strip()
    if file_ext == 'jpeg':
      file_ext = 'jpg'
    
    file_name = f'{img_ids[i]}.{file_ext}'
    file_path = path.join(DIR_PATH, file_name)

    file_obj.save(file_path)

    if file_ext == 'png':
      new_path = img_transforms.png_to_jpg(file_path)
      remove(file_path)
      file_path = new_path
    
    img_transforms.square_crop(file_path, MAX_IMG_SIZE)

  return f'successfully uploaded {batch_size} imgages'

@app.route('/getImages', methods=['GET'])
@cross_origin()
def get_images():
  img_count = request.args.get('img_count', type=int)
  img_ids = [request.args.get(f'id_{i}') for i in range(img_count)]

  print(img_ids)

  if not img_ids or len(img_ids) == 0:
    return 'bad request', 400

  images = [
    img_transforms.to_base64(
      path.join(DIR_PATH, f'{id}.jpg')
    )
      for id in img_ids
  ]
  
  return {
    'images': images,
  }

if __name__ == '__main__':
  app.run(port=PORT)