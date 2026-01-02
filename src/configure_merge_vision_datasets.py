"""
A script to configure the datasets downloaded by download-merge-vision-datasets.sh.

Note that some files must be manually downloaded, as specified in the docs at the top of the download script.

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
print("\nProcessing SUN397...")

sun_src_dir = base_dir / "sun397"  # replace with the path to your dataset
sun_dst_dir = base_dir / "sun397"  # replace with the path to the output directory


def copy_sun_split_files(split_file, downloaded_data_path, dst_dir):
    if dst_dir.exists():
        print(f"Skipping {split_file.name} because {dst_dir} already exists.")
        return

    print(f"Copying files from {split_file.name} into {dst_dir}...")
    with open(split_file, "r") as file:
        lines = file.readlines()

    for line in tqdm(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split("/")[:-1])[1:]
        filename = input_path.split("/")[-1]
        dst_class_dir = dst_dir / final_folder_name

        if not dst_class_dir.exists():
            dst_class_dir.mkdir(parents=True, exist_ok=True)

        src_path = downloaded_data_path / input_path[1:]
        dst_path = dst_class_dir / filename
        shutil.copy(src_path, dst_path)


copy_sun_split_files(
    sun_src_dir / "Partitions" / "Training_01.txt",
    sun_src_dir / "SUN397",
    sun_dst_dir / "train"
)
copy_sun_split_files(
    sun_src_dir / "Partitions" / "Testing_01.txt",
    sun_src_dir / "SUN397",
    sun_dst_dir / "val"
)


### PROCESS EuroSAT_RGB DATASET
print("\nProcessing EuroSAT...")

eurosat_src_dir = base_dir / "eurosat" / "2750"  # replace with the path to your dataset
eurosat_dst_dir = base_dir / "eurosat" / "splits"  # replace with the path to the output directory


def create_eurosat_directory_structure(dst_dir, classes):
    for dataset in ['train', 'val', 'test']:
        path = dst_dir / dataset
        path.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            (path / cls).mkdir(parents=True, exist_ok=True)


def split_eurosat(src_dir, dst_dir, classes, val_size=270, test_size=270):
    # First do a basic existence test to see if we should skip.
    test_dest_dir = dst_dir / "train" / classes[0]
    if test_dest_dir.is_dir() and next(test_dest_dir.iterdir(), None):
        print(f"Skipping {src_dir} because {test_dest_dir} appears to be already populated.")
        return

    print(f"Creating random split from {src_dir} to {dst_dir}...")
    for cls in tqdm(classes):
        class_path = src_dir / cls
        images = list(class_path.iterdir())
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for split, imgs in tqdm(zip(["train", "val", "test"], [train_images, val_images, test_images]),
                                desc=f"class: {cls}"):
            for src_path in tqdm(imgs, desc=f"{split} files"):
                dst_path = dst_dir / split / cls / src_path.name
                shutil.copy(src_path, dst_path)


classes = [d.name for d in eurosat_src_dir.iterdir() if d.is_dir()]
create_eurosat_directory_structure(eurosat_dst_dir, classes)
split_eurosat(eurosat_src_dir, eurosat_dst_dir, classes)


# PROCESS RESISC45 DATASET
print("\nProcessing RESISC45...")

resisc_src_dir = base_dir / "resisc45"  # replace with the path to your dataset
resisc_dst_dir = base_dir / "resisc45"  # replace with the path to the output directory


def copy_resisc_split_files(src_dir, dst_dir, split):
    split_file = src_dir / f"resisc45-{split}.txt"
    dst_dir = dst_dir / split
    if dst_dir.exists():
        print(f"Skipping {split_file.name} because {dst_dir} already exists.")
        return

    print(f"Copying files from {split_file.name} into {dst_dir}...")
    with open(split_file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        filename = line.strip()
        class_name = "_".join(filename.split("_")[:-1])
        dst_class_dir = dst_dir / class_name

        if not dst_class_dir.exists():
            dst_class_dir.mkdir(parents=True, exist_ok=True)

        src_path = src_dir / "NWPU-RESISC45" / class_name / filename
        dst_path = dst_class_dir / filename
        shutil.copy(src_path, dst_path)


for split in ["train", "val", "test"]:
    copy_resisc_split_files(resisc_src_dir, resisc_dst_dir, split)
