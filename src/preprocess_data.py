import pandas as pd
import random
import os
import glob

df = pd.read_csv(
    "/home/koushik/workspace/Projects/MLOPS/data/raw/lgg-mri-segmentation/kaggle_3m/data.csv"
)


data_map = []
for directory in glob.glob(
    "/home/koushik/workspace/Projects/MLOPS/data/raw/lgg-mri-segmentation/kaggle_3m/*"
):
    try:
        dir_name = directory.split("/")[-1]
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            data_map.extend([dir_name, image_path])
    except Exception as e:
        print(e)

new_df = pd.DataFrame({"patient_id": data_map[::2], "path": data_map[1::2]})

normal_images = new_df[~new_df["path"].str.contains("mask")]
mask_images = new_df[new_df["path"].str.contains("mask")]


# Sorting the Images and mask so as to form one-to-one correspondence
common_file_length = (
    123  # Any file with its absolute file path has minimum characters in the file name
)
normal_img_file_length = 4  # Normal file has only .tif common at its end, hence the value(len(.tif)) equal to 4
mask_img_file_length = 9  # mask files has _mask.tif at their end making its length(len(_mask.tif)) equal to 9

# Data sorting
imgs = sorted(
    normal_images["path"].values,
    key=lambda x: int(x[common_file_length:-normal_img_file_length]),
)
masks = sorted(
    mask_images["path"].values,
    key=lambda x: int(x[common_file_length:-mask_img_file_length]),
)

# x[common_file_length:-normal_img_file_length] is picking up the number stored after "TCGA_DU_7010_19860307_" and before ".tif" from a given Image x

# sanity check
idx = random.randint(0, len(imgs) - 1)
print("Path to the Image:", imgs[idx])
print("\nPath to the Mask:", masks[idx])

# Final dataframe
final_df = pd.DataFrame(
    {
        "patient_id": normal_images.patient_id.values,
        "image_path": imgs,
        "mask_path": masks,
    }
)

train_df = final_df.iloc[:3000, :]
val_df = final_df.iloc[3000:, :]

train_df.to_csv(
    "/home/koushik/workspace/Projects/MLOPS/data/preprocessed/train.csv", index=False
)
val_df.to_csv(
    "/home/koushik/workspace/Projects/MLOPS/data/preprocessed/eval.csv", index=False
)
