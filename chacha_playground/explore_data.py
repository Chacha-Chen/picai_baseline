from pathlib import Path
import argparse
import json
import os
from pathlib import Path

import SimpleITK as sitk
from picai_prep import MHA2nnUNetConverter
from picai_prep.data_utils import atomic_image_write
from picai_prep.examples.mha2nnunet.picai_archive import \
    generate_mha2nnunet_settings
from tqdm import tqdm
from picai_baseline.splits.picai_debug import nnunet_splits


# set paths
parser = argparse.ArgumentParser()
parser.add_argument("--workdir", type=str, default=os.environ.get("workdir", "/workdir"),
                    help="Path to the working directory (default: /workdir, or the environment variable 'workdir')")
parser.add_argument("--inputdir", type=str, default=os.environ.get("inputdir", "/input"),
                    help="Path to the input dataset (default: /input, or the environment variable 'inputdir')")
parser.add_argument("--imagesdir", type=str, default="images",
                    help="Path to the images, relative to --inputdir (default: /input/images)")
parser.add_argument("--labelsdir", type=str, default="labels",
                    help="Path to the labels, relative to --inputdir (root of picai_labels) (default: /input/picai_labels)")
parser.add_argument("--spacing", type=float, nargs="+", required=False,
                    help="Spacing to preprocess images to. Default: keep as-is.")
parser.add_argument("--matrix_size", type=int, nargs="+", required=False,
                    help="Matrix size to preprocess images to. Default: keep as-is.")
parser.add_argument("--preprocessing_kwargs", type=str, required=False,
                    help='Preprocessing kwargs to pass to the MHA2nnUNetConverter. " + \
                         "E.g.: `{"crop_only": true}`. Must be valid json.')
try:
    args = parser.parse_args()
except Exception as e:
    print(f"Parsing all arguments failed: {e}")
    print("Retrying with only the known arguments...")
    args, _ = parser.parse_known_args()


# parse paths
workdir = Path(args.workdir)
inputdir = Path(args.inputdir)
imagesdir = Path(inputdir / args.imagesdir)
labelsdir = Path(inputdir / args.labelsdir)

# settings
task = "Task2203_picai_baseline"

# relative paths
annotations_dir_human_expert = labelsdir / "csPCa_lesion_delineations/human_expert/resampled/"
annotations_dir_ai_derived = labelsdir / "csPCa_lesion_delineations/AI/Bosma22a/"
annotations_dir = labelsdir / "csPCa_lesion_delineations/combined/"
mha2nnunet_settings_path = workdir / "mha2nnunet_settings" / (task + ".json")
nnUNet_raw_data_path = workdir / "nnUNet_raw_data"
nnUNet_task_dir = nnUNet_raw_data_path / task
nnUNet_dataset_json_path = nnUNet_task_dir / "dataset.json"
nnUNet_splits_path = nnUNet_task_dir / "splits.json"

#%%

with open('/net/scratch/chacha/workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json') as fp:
    mha2nnunet_settings = json.load(fp)

# with open('/net/scratch/chacha/workdir/nnUNet_raw_data/Task2201_picai_baseline/splits.json')