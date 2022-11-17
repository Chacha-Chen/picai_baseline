#  Run a basic script for testing
import argparse
import json
import os
from pathlib import Path
from subprocess import check_call
from picai_baseline.splits.picai_debug import nnunet_splits
# from nnunet_wrapper_chacha import nnunet_wrapper
import os
from subprocess import call
import signal
import threading
import time
import logging
import socket
from picai_baseline.nnunet.eval import evaluate
# import torch
from picai_eval import Metrics


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# print(logger)

def is_on_slurm():
    return os.environ.get("SLURM_JOB_ID") is not None

def schedule_death(seconds, verbose=False):
    logger.info(f"scheduling death after {seconds}s")
    def f():
        death = time.time() + seconds
        while time.time() < death:
            if verbose:
                logger.info(f"Beep...")
            sleep_interval = max(0, min(600, death - time.time()))
            time.sleep(sleep_interval)
        
        logger.info(f"time to die...")
        logging.shutdown()
        print("signal.sigusr1", signal.SIGUSR1)
        os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=f, daemon=True).start()

def slurm_sigusr1_handler_fn(signum, frame) -> None:
    logger.info(f"received signal {signum}")
    job_id = os.environ["SLURM_JOB_ID"]
    cmd = ["scontrol", "requeue", job_id]

    logger.info(f"requeing job {job_id}...")
    try:
        result = call(cmd)
    except FileNotFoundError:
        joint_cmd = [str(x) for x in cmd]
        result = call(" ".join(joint_cmd), shell=True)

    if result == 0:
        logger.info(f"requeued exp {job_id}")
    else:
        logger.info("requeue failed")

def setup_slurm():
    if not is_on_slurm():
        logger.info("not running in slurm, this job will run until it finishes.")
        return
    logger.info("running in slurm, ready to requeue on SIGUSR1.")
    # print("running in slurm, ready to requeue on SIGUSR1.")
    signal.signal(signal.SIGUSR1, slurm_sigusr1_handler_fn)
    # slurm not sending the signal, so sending it myself
    time_to_live = 14300  #14300 # just a bit less than 4 hrs
    schedule_death(time_to_live)






def main():
    """Train nnU-Net semi-supervised model."""
    parser = argparse.ArgumentParser()
    # input data and model directories
    parser.add_argument('--task', type=str, default="Task2203_picai_baseline")

    parser.add_argument('--workdir', type=str, default=os.environ.get('workdir',"/workdir"))
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('imagesdir', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('labelsdir', "/input/picai_labels"))
    # parser.add_argument('--scriptsdir', type=str, default=os.environ.get('SM_CHANNEL_SCRIPTS', "/scripts"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('outputdir', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default=os.environ.get('checkpointsdir', "/checkpoints"))
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--no_debug', action='store_true', help='Debug training --- overwrite validation with the same training data')
    parser.add_argument('--do_train', action='store_true', help='Do training')
    parser.add_argument('--do_eval', action='store_true', help='Do evaluation')
    # parser.add_argument('--nnUNet_n_proc_DA', type=int, default=None)
    args, _ = parser.parse_known_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    taskname = args.task
    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    # scripts_dir = Path(args.scriptsdir)
    checkpoints_dir = Path(args.checkpointsdir)
    nnUNet_splits_path = workdir / f"nnUNet_raw_data/{taskname}/splits.json"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    nnUNet_splits_path.parent.mkdir(parents=True, exist_ok=True)

    # set environment variables
    

    # extract scripts
    # with zipfile.ZipFile(scripts_dir / "code.zip", 'r') as zf:
    #     zf.extractall(workdir)
    # local_scripts_dir = workdir / "code"

    # descibe input data
    logging.info(f"workdir: {workdir}")
    logging.info(f"images_dir: {images_dir}")
    logging.info(f"labels_dir: {labels_dir}")
    logging.info(f"output_dir: {output_dir}")

    # install modified nnU-Net
    # print("Installing modified nnU-Net...")
    # cmd = [
    #     "pip",
    #     "install",
    #     "-e",
    #     str(local_scripts_dir / "nnunet"),
    # ]
    # check_call(cmd)

    # save cross-validation splits to disk
    if not args.no_debug:
        logging.info("Debugging mode")
        folds = range(1)  # range(5) for 5-fold cross-validation 
        # nnunet_splits[0]['val'] = nnunet_splits[0]['train']
    else:
        folds = range(5)
    
    with open(nnUNet_splits_path, "w") as fp:
        print("writing to ", nnUNet_splits_path)
        json.dump(nnunet_splits, fp)

    # Convert MHA Archive to nnU-Net Raw Data Archive
    # Also, we combine the provided human-expert annotations with the AI-derived annotations.
    # print("Preprocessing data...")
    # cmd = [
    #     "python",
    #     (local_scripts_dir / "picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py").as_posix(),
    #     "--workdir", workdir.as_posix(),
    #     "--imagesdir", images_dir.as_posix(),
    #     "--labelsdir", labels_dir.as_posix(),
    # ]
    # check_call(cmd)

    # Train models
    # if args.nnUNet_n_proc_DA is not None:
    #     os.environ["nnUNet_n_proc_DA"] = str(args.nnUNet_n_proc_DA)

    
    ## TODO i dont need to run sequentially
    if args.do_train:
        for fold in folds:
            print(f"Training fold {fold}...")
            cmd = [
                # f"CUDA_VISIBLE_DEVICES={args.gpu}",
                "python", Path(os.environ["parent_dir"] + "/picai_baseline/src/picai_baseline/nnunet/training_docker/nnunet_wrapper.py").as_posix(),
                "plan_train", str(taskname), workdir.as_posix(),
                "--trainer_kwargs", "{\"max_num_epochs\":20}",
                "--results", checkpoints_dir.as_posix(),
                "--trainer", "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
                "--fold", str(fold),
                "--custom_split", str(nnUNet_splits_path),
            ]
            # nnunet_wrapper("plan_train", str(taskname), workdir.as_posix())
            check_call(cmd)


    # Inference 
    if args.do_eval:
        for fold in folds:
            print(f"Inference fold {fold}...")
            cmd = [
                "python", Path(os.environ["parent_dir"] + "/picai_baseline/src/picai_baseline/nnunet/training_docker/nnunet_wrapper.py").as_posix(),
                "predict", str(taskname), workdir.as_posix(),
                "--trainer", "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
                "--fold", str(fold),
                "--checkpoint", "model_best",
                "--results", checkpoints_dir.as_posix(),
                "--input", str(workdir / "nnUNet_raw_data"/ taskname / "imagesTr"),
                "--output", str(workdir / "results/nnUNet/3d_fullres"/ taskname/ f"nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/picai_pubtrain_predictions_model_best"),
                "--store_probability_maps"
            ]
            check_call(cmd)
            # nnunet predict Task2201_picai_baseline \
            # --trainer nnUNetTrainerV2_Loss_FL_and_CE_checkpoints \
            # --fold 0 --checkpoint model_best \
            # --results /workdir/results \
            # --input /input/images/ \
            # --output /output/predictions \
            # --store_probability_maps
            
            

            # evaluate
            metrics = evaluate(
                task = taskname,
                workdir = workdir.as_posix(),
                folds=[fold],
                num_parallel_calls = 1
            )

            fold = 0
            checkpoint = "model_best"
            threshold = "dynamic"
            # # metric_path = str(workdir / "nnUNet_preprocessed")
            metrics = Metrics(f"{workdir}/results/nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/metrics-{checkpoint}-{threshold}.json")
            print(f"PI-CAI ranking score: {metrics.score:.4f} (50% AUROC={metrics.auroc:.4f} + 50% AP={metrics.AP:.4f})")

            
    # python /data/chacha/picai_baseline/src/picai_baseline/nnunet/eval.py --task=Task2203_picai_baseline --workdir=/data/chacha/picai_data/workdir


    # Export trained models
    # for fold in folds:
    #     src = checkpoints_dir / f"nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model"
    #     dst = output_dir / f"picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model"
    #     dst.mkdir(parents=True, exist_ok=True)
    #     shutil.copy(src, dst)

    #     src = checkpoints_dir / f"nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model.pkl"
    #     dst = output_dir / f"picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/fold_{fold}/model_best.model.pkl"
    #     shutil.copy(src, dst)

    # shutil.copy(checkpoints_dir / f"nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/plans.pkl",
    #             output_dir / f"picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/{taskname}/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/plans.pkl")


if __name__ == '__main__':
    setup_slurm()
    # print(is_on_slurm())
    if socket.gethostname() != 'bingo':
        parent_dir = '/net/scratch/chacha'
    else:
        parent_dir = '/data/chacha'


    workdir = Path(parent_dir +'/picai_data/workdir')
    os.environ["parent_dir"] = parent_dir
    os.environ["prepdir"] = str(workdir / "nnUNet_preprocessed")
    os.environ["workdir"] = str(workdir)
    os.environ["imagesdir"] = parent_dir + '/picai_data/input/images'
    os.environ["labelsdir"] = parent_dir +'/picai_data/input/labels'
    os.environ["outputdir"] = parent_dir +'/picai_data/output'
    os.environ["checkpointsdir"] = str(workdir / "results")
    os.environ["nnUNet_preprocessed"] = str(workdir / "nnUNet_preprocessed")
    os.environ['nnUNet_raw_data_base'] = workdir.as_posix()

    # export nnUNet_raw_data_base="/net/scratch/chacha/picai_data/workdir"
    # export nnUNet_preprocessed="/net/scratch/chacha/picai_data/workdir/nnUNet_preprocessed"
    # export RESULTS_FOLDER="/net/scratch/chacha/picai_data/workdir/results"

    # check_call("nnUNet_convert_decathlon_task -i /net/scratch/chacha/picai_data/Task05_Prostate")
    # check_call("nnUNet_download_pretrained_model Task005_Prostate")
    # check_call("nnUNet_print_pretrained_model_info Task005_Prostate")
    main()
