"""
A script to configure the datasets downloaded by download-merge-vision-datasets.sh.

These are some of the datasets used by the Task Vectors paper, and subsequent papers, to evaluate model merging. The
rest of the datasets used by this line of work can be acquired directly through Torchvision, but these few require
manual setup. These scripts automate most of the setup.
Task Vectors: https://github.com/mlfoundations/task_vectors/
Dataset issues: https://github.com/mlfoundations/task_vectors/issues/1
"""
import random
import shutil
from pathlib import Path

from tqdm import tqdm


base_dir = Path("data")


### PROCESS SUN397 DATASET
print("Processing SUN397...")

downloaded_data_path = base_dir / "sun397"  # replace with the path to your dataset
sun_dir = base_dir / "sun397"  # replace with the path to the output directory


def process_dataset(txt_file, downloaded_data_path, output_folder):
    print(f"Moving files from {txt_file.name} into {output_folder}...")
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for line in tqdm(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split('/')[:-1])[1:]
        filename = input_path.split('/')[-1]
        output_class_folder = output_folder / final_folder_name

        if not output_class_folder.exists():
            output_class_folder.mkdir()

        full_input_path = downloaded_data_path / input_path[1:]
        output_file_path = output_class_folder / filename
        # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
        # exit()
        shutil.copy(full_input_path, output_file_path)


process_dataset(
    downloaded_data_path / 'Partitions' / 'Training_01.txt',
    downloaded_data_path / 'SUN397',
    sun_dir / "train"
)
process_dataset(
    downloaded_data_path / 'Partitions' / 'Testing_01.txt',
    downloaded_data_path / 'SUN397',
    sun_dir / "val"
)


### PROCESS EuroSAT_RGB DATASET

src_dir = base_dir / "eurosat" / "2750"  # replace with the path to your dataset
dst_dir = base_dir / "eurosat_splits"  # replace with the path to the output directory


def create_directory_structure(dst_dir, classes):
    for dataset in ['train', 'val', 'test']:
        path = dst_dir / dataset
        path.mkdir(exist_ok=True)
        for cls in classes:
            (path / cls).mkdir(exist_ok=True)


def split_dataset(dst_dir, src_dir, classes, val_size=270, test_size=270):
    for cls in classes:
        class_path = src_dir / cls
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for img in train_images:
            src_path = class_path / img
            dst_path = dst_dir / "train" / cls / img
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            # break
        for img in val_images:
            src_path = class_path / img
            dst_path = dst_dir / "val" / cls / img
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            # break
        for img in test_images:
            src_path = class_path / img
            dst_path = dst_dir / "test" / cls / img
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            # break


classes = [d for d in src_dir.iterdir() if d.is_dir()]
create_directory_structure(dst_dir, classes)
split_dataset(dst_dir, src_dir, classes)
