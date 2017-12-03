import numpy as np
from PIL import ImageDraw
from PIL import Image


def load_images_labels_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    images_names = f.readlines()
    images_names = [x.split(None, 1)[1] for x in images_names]
    images_names = [x.strip('\n').strip(None) for x in images_names]
    return images_names


def load_image_data(path_voc, class_object):
    print("load images" + path_voc)
    image_names = np.array(load_images_names_in_data_set('aeroplane_trainval', path_voc))
    labels = load_images_labels_in_data_set('aeroplane_trainval', path_voc)
    image_names_class = []
    for i in range(len(image_names)):
        if labels[i] == class_object:
            image_names_class.append(image_names[i])
    image_names = image_names_class
    images = get_all_images(image_names, path_voc)
    print("total image:%d" % len(image_names))
    return image_names, images


def load_images_names_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    return [x.split(None, 1)[0] for x in image_names]


def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        img = Image.open(string)
        images.append(np.array(img))
    return images