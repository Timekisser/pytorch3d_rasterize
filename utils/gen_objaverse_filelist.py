import os
import wget
import pandas as pd
from tqdm import tqdm
import objaverse
output_dir = "data/Objaverse"

cap3d_filelist_path = os.path.join(output_dir, "filelist", "Cap3D_automated_Objaverse_highquality.csv")

if not os.path.exists(cap3d_filelist_path):
    os.system(f"wget https://hf-mirror.com/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_highquality.csv?download=true -O {cap3d_filelist_path}")

cap3d_filelist = pd.read_csv(cap3d_filelist_path, header=None, names=["uid", "description"])
cap3d_uids = cap3d_filelist["uid"].to_list()

objaverse._VERSIONED_PATH = "data/Objaverse/objaverse"
objaverse_uids = objaverse.load_uids()
objaverse_glbs = objaverse.load_objects(objaverse_uids, 24)

total_length = 8000000
uids = []
for uid in tqdm(cap3d_uids):
    # filename = f"{uid[0]}/{uid}"
    filename = objaverse_glbs[uid].split("/")[4] + "/" + uid
    if os.path.exists(os.path.join(output_dir, "pointcloud_20w", filename, "pointcloud.npz")):
        uids.append(filename)
        if len(uids) > total_length:
            break

length = min(len(uids), total_length)
train_length = int(0.95 * length)
all_list = os.path.join(output_dir, "filelist", 'all.txt')
train_list = os.path.join(output_dir, "filelist", 'train.txt')
eval_list = os.path.join(output_dir, "filelist", 'test.txt')
with open(all_list, "w") as f:
    for filename in uids[:length]:
        f.write(filename)
        f.write('\n')
with open(train_list, "w") as f:
    for filename in uids[:train_length]:
        f.write(filename)
        f.write('\n')
with open(eval_list, "w") as f:
    for filename in uids[train_length:length]:
        f.write(filename)
        f.write('\n')
