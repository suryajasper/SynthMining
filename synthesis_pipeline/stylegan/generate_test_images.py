import os
import requests
import argparse
from PIL import Image

# Define the URL and the folder to save the images
url = 'https://thispersondoesnotexist.com/'
folder_path = 'downloaded_images/'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Function to download images
def download_images(num_images, size):
    for i in range(num_images):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Save the image to the specified folder
            file_path = os.path.join(folder_path, f'image_{i}.jpg')
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Open the image and resize if necessary
            if size != 1024:
                img = Image.open(file_path)
                img = img.resize((size, size), Image.ANTIALIAS)
                img.save(file_path)
            
            print(f"Image {i} saved.")
        else:
            print(f"Failed to download image {i}.")

    print("All images downloaded successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from a URL and optionally resize them.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to download')
    parser.add_argument('--size', type=int, default=1024, help='Size to scale the images (a power of 2, e.g., 64, 128, 512)')

    args = parser.parse_args()

    download_images(args.num_images, args.size)
