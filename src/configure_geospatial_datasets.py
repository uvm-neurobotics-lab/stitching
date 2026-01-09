"""
A script to configure the datasets downloaded by download-geospatial-datasets.sh.
"""
import random
import shutil
from pathlib import Path

from tqdm import tqdm


base_dir = Path("data")


def split_dataset(src_dir, dst_dir, val_frac=0.0, test_frac=0.2):
    """
    Create a copy of the ImageFolder-compatible dataset at `src_dir`, split into train, val, and test. The copy will be
    made at `dst_dir` (which can also be the same dir, if desired). Note that by default only a test split will be
    created, without a validation set (`val_frac` is set to 0.0).

    Args:
        src_dir: The folder containing per-class subfolders.
        dst_dir: The folder at which train/, val/, and test/ folders should be stored.
        val_frac: What fraction of the dataset to use as validation (can be 0.0 for no validation).
        test_frac: What fraction of the dataset to use as test (can be 0.0 for no test).
    """
    # First do a basic existence test to see if we should skip.
    test_dest_dir = dst_dir / "train"
    if test_dest_dir.is_dir() and next(test_dest_dir.iterdir(), None):
        print(f"Skipping {src_dir} because {test_dest_dir} appears to be already populated.")
        return

    class_dirs = [d for d in src_dir.iterdir() if d.is_dir()]

    print(f"Creating random split from {src_dir} to {dst_dir}...")
    for class_dir in tqdm(class_dirs):
        images = list(class_dir.iterdir())
        random.shuffle(images)

        val_size = int(len(images) * val_frac)
        test_size = int(len(images) * test_frac)
        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for split, imgs in tqdm(zip(["train", "val", "test"], [train_images, val_images, test_images]),
                                desc=f"class: {class_dir.name}"):
            dst_class_dir = dst_dir / split / class_dir.name
            if imgs:  # Skip directory creation if empty.
                dst_class_dir.mkdir(parents=True, exist_ok=True)
            for src_path in tqdm(imgs, desc=f"{split} files"):
                dst_path = dst_class_dir / src_path.name
                shutil.copy(src_path, dst_path)


### PROCESS AID DATASET
print("\nProcessing AID...\n")
aid_src_dir = base_dir / "AID"  # replace with the path to your dataset
aid_dst_dir = base_dir / "AID"  # replace with the path to the output directory
split_dataset(aid_src_dir, aid_dst_dir)


### PROCESS UCM DATASET
print("\nProcessing UCM Land Use...\n")
ucm_src_dir = base_dir / "UCMerced_LandUse" / "Images"  # replace with the path to your dataset
ucm_dst_dir = base_dir / "UCMerced_LandUse"  # replace with the path to the output directory
split_dataset(ucm_src_dir, ucm_dst_dir)
