import pandas as pd
import yaml
import glob
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def get_file_row(path):
    path_no_ext, ext = os.path.splitext(path)
    filename = os.path.basename(path)

    patient_id = "_".join(filename.split("_")[:3])

    return [patient_id, path, f"{path_no_ext}_mask{ext}"]


def preprocessing(config):
    files_dir = "data/raw/lgg-mri-segmentation/kaggle_3m/"
    file_paths = glob.glob(f"{files_dir}/*/*[0-9].tif")

    csv_path = "data/raw/lgg-mri-segmentation/kaggle_3m/data.csv"
    df = pd.read_csv(csv_path)

    imputer = SimpleImputer(strategy="most_frequent")
    print(list(df.columns))
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    filenames_df = pd.DataFrame(
        (get_file_row(filename) for filename in file_paths),
        columns=["Patient", "image_filename", "mask_filename"],
    )
    df = pd.merge(df, filenames_df, on="Patient")

    train_df, test_df = train_test_split(df, test_size=0.3)
    test_df, valid_df = train_test_split(test_df, test_size=0.5)

    train_df.to_csv(config["dataset"]["train_csv"], index=False)
    valid_df.to_csv(config["dataset"]["valid_csv"], index=False)
    test_df.to_csv(config["dataset"]["test_csv"], index=False)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    preprocessing(config)
