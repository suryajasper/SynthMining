import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

# ** WEIGHT INITIALIZATION **
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def to_tensor(image_set, image_size, test_size, to3colorlayer) -> list:
    """
    convert an PIL.Image.Image set into a list of 3 layer, will
    """
    if not isinstance(image_set[0], Image.Image):
        raise ValueError("image_set is not iterable PIL.Image i. Fix me")
    tensors = []
    for i in image_set[:test_size]:
        if to3colorlayer:  # when feeding b&w or gray scale but you want 3 layer (bc your model is like that)
            i = i.convert('RGB')
        scaled = tf.resize(i, size=image_size)  # scale image
        cropped = tf.center_crop(scaled, output_size=min(scaled.size))  # use min dimension to crop

        # transform from image to tensor
        toTensor = transforms.Compose([transforms.PILToTensor()])
        out_tensor = toTensor(cropped)
        tensors.append(out_tensor)
    return tensors


def to_batches(dataset, batch_size):
    """
    take entire data set and pack into training batch
    """
    img_batches = []
    shift = 0
    while shift < len(dataset):
        stride = min(batch_size, len(dataset) - shift)
        remainder = batch_size - stride

        img_size = dataset[0].size()[1]

        batch = torch.Tensor(batch_size, 3, img_size, img_size)

        batch_list = dataset[shift: shift + stride] + dataset[:remainder]

        for i, tensor in enumerate(batch_list):
            batch[i] = tensor

        img_batches.append(
            batch
        )
        shift += stride
    return img_batches

