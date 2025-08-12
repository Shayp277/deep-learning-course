import kagglehub
import os

def download_data(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if len(os.listdir(data_dir)) != 0:
        print("data dir not empty")
        return

    # Download latest version
    path = kagglehub.dataset_download("janboubiabderrahim/vehicle-sounds-dataset")

    print("Path to dataset files:", path)