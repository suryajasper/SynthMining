from PIL import Image, ImageOps
from os import path
import base64

# takes a list of files to convert
def square_crop(file_path, image_size):
  img = Image.open(file_path)
  width, height = img.size

  # crop image to square
  if width > height:
    img = img.crop((
      (width - height) / 2, 0,
      (width + height) / 2, height,
    ))
  else:
    img = img.crop((
      0, (height - width) / 2,
      width, (height + width) / 2,
    ))

  # resize image to fit specified dimensions
  img = img.resize((image_size, image_size))
  
  # save image
  img.save(file_path)

def png_to_jpg(file_path):
  file_name = file_path[:-4]

  img = Image.open(f'{file_name}.png')
  rgb_img = img.convert('RGB')
  rgb_img.save(f'{file_name}.jpg')

  return f'{file_name}.jpg'

def to_base64(file_name):
  data = ''
  with open(file_name , "rb") as image_file :
    data = base64.b64encode(image_file.read())
  return data.decode('utf-8')