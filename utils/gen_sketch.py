import cv2
import os
from tqdm import tqdm
def get_sketch(image):
    edges = cv2.Canny(image=image, threshold1=20, threshold2=180)
    edges = cv2.GaussianBlur(edges, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.bitwise_not(edges)
    edges[edges < 255] = 0
    return edges

def get_filenames(filelist):
    with open(filelist, 'r') as fid:
        lines = fid.readlines()
    filenames = [line.split()[0] for line in lines]
    return filenames

image_dir = "data/ShapeNet/image_1024"
save_dir = "data/ShapeNet/sketch_1024"

filenames = get_filenames("data/ShapeNet/filelist/train_im_5.txt") + get_filenames("data/ShapeNet/filelist/test_im_5.txt")

for filename in tqdm(filenames):
    for image_name in os.listdir(os.path.join(image_dir, filename)):
        image_path = os.path.join(image_dir, filename, image_name)
        image = cv2.imread(image_path)
        sketch = get_sketch(image)
        save_path = os.path.join(save_dir, filename, image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, sketch)
