import os
import pickle
from visulization import vis_all

ROOT_DIR = '../data'

IMAGE_PATH = ROOT_DIR + '/image/'
RESULT_PATH = ROOT_DIR + '/result/'
all_image_path = os.listdir(IMAGE_PATH)

for im_name in all_image_path:
    image_name = im_name.split('.')[0]
    result_path = os.path.join(RESULT_PATH, image_name)

    with open(os.path.join(result_path, 'sampler.pickle'), 'rb') as f:
        sampler = pickle.load(f, encoding='latin1')

    vis_all(sampler, IMAGE_PATH + im_name, result_path, save_image=False)
    pass

