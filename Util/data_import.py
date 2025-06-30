import kagglehub
import os

def download_data():
    if not os.path.exists('../data'):
        os.makedirs('../data')

    if len(os.listdir('../data')) != 0:
        print("data dir not empty")
        return

    # Download latest version
    path = kagglehub.dataset_download("janboubiabderrahim/vehicle-sounds-dataset")

    print("Path to dataset files:", path)