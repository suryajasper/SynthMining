import os

folder_path = "img_align_celeba"
output_file = "trainer_set.txt"

with open(output_file, "w") as f:
    for filename in os.listdir(folder_path):
        f.write(filename + "\n")