import os
import image_transform as it # adarsh code


# user it.resizer(filename)
def to_tensor(fileDir):
    tensors = []
    for file in os.listdir(fileDir):
        tensors.append(it.resizer(file))
    return tensors

def to_batches(dataSet, batch_size):
    img_batches = []
    shift = 0
    while shift < len(dataSet):
        stride = min(batch_size, len(dataSet)-shift)
        remainder = batch_size - stride
        img_batches.append(
            dataSet[shift : shift+stride] + 
            dataSet[:remainder]
        )
        shift += stride
    return img_batches


