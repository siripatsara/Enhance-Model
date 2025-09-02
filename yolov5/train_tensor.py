# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Enhanced YOLOv5 Training with TensorBoard Visualization
Train a YOLOv5 model on a custom dataset with comprehensive TensorBoard logging.

Features:
- Box regression loss tracking
- Objectness loss tracking  
- Box classification loss tracking
- Segmentation loss tracking (for seg models)
- Learning rate monitoring
- Accuracy plots (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
- Precision-Recall curves with mAP calculation at IoU@0.5

Usage - Single-GPU training:
    $ python train_tensor.py --data dataset_allBB/data.yaml --weights yolov5s.pt --img 640 --epochs 50 --batch-size 16
    $ python train_tensor.py --data dataset_allBB/data.yaml --weights yolov5s-seg.pt --img 640 --epochs 50 --batch-size 16

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train_tensor.py --data dataset_allBB/data.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
TensorBoard: https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)
from utils.plots import plot_evolve, plot_images, plot_lr_scheduler, plot_results
from utils.metrics import fitness, ap_per_class
from utils.loss import ComputeLoss
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loggers import LOGGERS, Loggers
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.downloads import attempt_download, is_url
from utils.dataloaders import create_dataloader
from utils.callbacks import Callbacks
from utils.autobatch import check_train_batch_size
from utils.autoanchor import check_anchors
from models.yolo import Model
from models.experimental import attempt_load
import val as validate  # for end-of-epoch mAP
from ultralytics.utils.patches import torch_load
import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train_with_tensorboard(hyp, opt, device, callbacks):
    """
    Enhanced YOLOv5 training function with comprehensive TensorBoard logging.

    Logs the following metrics to TensorBoard:
    - Box regression loss
    - Objectness loss
    - Classification loss
    - Segmentation loss (if using seg model)
    - Learning rate
    - Training and validation metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
    - Model graph visualization
    - Training images and predictions

    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.

    Returns:
        tuple: Training results (precision, recall, mAP@0.5, mAP@0.5:0.95, val_loss(box, obj, cls))
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Initialize TensorBoard Writer
    tb_writer = None
    if RANK in {-1, 0}:
        tb_writer = SummaryWriter(str(save_dir))
        LOGGER.info(f"🚀 TensorBoard initialized!")
        LOGGER.info(
            f"📊 Results will be saved to: {colorstr('bold', save_dir)}")
        LOGGER.info(f"📁 Model weights: {colorstr('bold', w)}")
        LOGGER.info(f"📈 TensorBoard logs: {colorstr('bold', save_dir)}")
        LOGGER.info(
            f"🌐 Start TensorBoard: tensorboard --logdir {save_dir.parent}")
        LOGGER.info(f"🔗 View at: http://localhost:6006/")

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") +
                ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(
        data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith(
        "coco/val2017.txt")  # COCO dataset
    is_seg = 'seg' in str(weights).lower()  # Check if using segmentation model

    # Model
    check_suffix(weights, ".pt")  # check weights
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            # download if not found locally
            weights = attempt_download(weights)
        # load checkpoint to CPU to avoid CUDA memory leak
        ckpt = torch_load(weights, map_location="cpu")
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get(
            "anchors")).to(device)  # create
        exclude = ["anchor"] if (cfg or hyp.get(
            "anchors")) and not resume else []  # exclude keys
        # checkpoint state_dict as FP32
        csd = ckpt["model"].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(),
                              exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get(
            "anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(
        freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(
        model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:

        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear

    # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Plot learning rate schedule to TensorBoard
    if plots and tb_writer and RANK in {-1, 0}:
        plot_lr_scheduler(optimizer, scheduler, epochs, save_dir)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(
                ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            if not opt.noautoanchor:
                # run AutoAnchor
                check_anchors(dataset, model=model,
                              thr=hyp["anchor_t"], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    # number of detection layers (to scale hyps)
    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Log model graph to TensorBoard
    if tb_writer and RANK in {-1, 0}:
        try:
            dummy_input = torch.zeros(1, 3, imgsz, imgsz).to(device)
            tb_writer.add_graph(model, dummy_input)
            LOGGER.info("Model graph logged to TensorBoard")
        except Exception as e:
            LOGGER.warning(f"Failed to log model graph to TensorBoard: {e}")

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    # number of warmup iterations, max(3 epochs, 100 iterations)
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")

    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"📁 All results will be saved to: {colorstr('bold', save_dir)}\n"
        f"📊 TensorBoard logging enabled - view metrics at http://localhost:6006/\n"
        f"🏃‍♂️ Starting training for {epochs} epochs..."
    )

    # Log hyperparameters to TensorBoard
    if tb_writer and RANK in {-1, 0}:
        for key, value in hyp.items():
            tb_writer.add_scalar(f'Hyperparameters/{key}', value, 0)

        # Log training configuration
        tb_writer.add_text('Config/Model', str(weights))
        tb_writer.add_text('Config/Dataset', str(data))
        tb_writer.add_text('Config/Classes', str(names))
        tb_writer.add_scalar('Config/Epochs', epochs, 0)
        tb_writer.add_scalar('Config/BatchSize', batch_size, 0)
        tb_writer.add_scalar('Config/ImageSize', imgsz, 0)
        tb_writer.add_scalar('Config/NumClasses', nc, 0)

    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(
                range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4 if is_seg else 3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem",
                    "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            # progress bar
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()

        # batch -------------------------------------------------------------
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run("on_train_batch_start")
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(
                    int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log to TensorBoard every N batches
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / \
                    (i + 1)  # update mean losses
                # (GB)
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )

                # Log batch-level metrics to TensorBoard
                if tb_writer and ni % 50 == 0:  # Log every 50 batches
                    step = ni
                    current_lr = optimizer.param_groups[0]["lr"]

                    # Loss logging
                    tb_writer.add_scalar("Loss/Box_Regression", mloss[0], step)
                    tb_writer.add_scalar("Loss/Objectness", mloss[1], step)
                    tb_writer.add_scalar("Loss/Classification", mloss[2], step)
                    if is_seg and len(mloss) > 3:
                        tb_writer.add_scalar(
                            "Loss/Segmentation", mloss[3], step)
                    tb_writer.add_scalar("Loss/Total", mloss.sum(), step)

                    # Learning rate and other metrics
                    tb_writer.add_scalar("Learning_Rate", current_lr, step)
                    tb_writer.add_scalar("GPU_Memory_GB", torch.cuda.memory_reserved(
                    ) / 1e9 if torch.cuda.is_available() else 0, step)
                    tb_writer.add_scalar("Batch_Size", targets.shape[0], step)

                # Log training images periodically
                if tb_writer and ni % 1000 == 0 and plots:  # Log every 1000 batches
                    try:
                        # Log a few training images with targets
                        f = save_dir / f"train_batch{ni}.jpg"
                        plot_images(imgs[:8], targets[:8], paths[:8], f, names)
                        if f.exists():
                            tb_writer.add_image(f"Train/Batch_{ni}",
                                                torch.from_numpy(plt.imread(f)).permute(2, 0, 1), ni)
                    except Exception as e:
                        LOGGER.warning(f"Failed to log training images: {e}")

                callbacks.run("on_train_batch_end", model, ni,
                              imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(
                model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Log epoch-level metrics to TensorBoard
            if tb_writer:
                epoch_step = epoch

                # Training losses
                tb_writer.add_scalar(
                    "Epoch/Train_Loss_Box", mloss[0], epoch_step)
                tb_writer.add_scalar(
                    "Epoch/Train_Loss_Objectness", mloss[1], epoch_step)
                tb_writer.add_scalar(
                    "Epoch/Train_Loss_Classification", mloss[2], epoch_step)
                if is_seg and len(mloss) > 3:
                    tb_writer.add_scalar(
                        "Epoch/Train_Loss_Segmentation", mloss[3], epoch_step)
                tb_writer.add_scalar(
                    "Epoch/Train_Loss_Total", mloss.sum(), epoch_step)

                # Validation metrics
                tb_writer.add_scalar("Metrics/Precision",
                                     results[0], epoch_step)
                tb_writer.add_scalar("Metrics/Recall", results[1], epoch_step)
                tb_writer.add_scalar("Metrics/mAP_0.5", results[2], epoch_step)
                tb_writer.add_scalar("Metrics/mAP_0.5:0.95",
                                     results[3], epoch_step)

                # Validation losses
                tb_writer.add_scalar("Epoch/Val_Loss_Box",
                                     results[4], epoch_step)
                tb_writer.add_scalar(
                    "Epoch/Val_Loss_Objectness", results[5], epoch_step)
                tb_writer.add_scalar(
                    "Epoch/Val_Loss_Classification", results[6], epoch_step)

                # Learning rate
                tb_writer.add_scalar("Epoch/Learning_Rate", lr[0], epoch_step)

                # Per-class mAP
                for j, c in enumerate(names.values()):
                    tb_writer.add_scalar(
                        f"mAP_per_class/{c}", maps[j], epoch_step)

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
                if tb_writer:
                    tb_writer.add_scalar("Best/Fitness", best_fitness, epoch)

            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals,
                          epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch,
                              final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            # broadcast 'stop' to all ranks
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(
            f"\n🎉 Training completed! {epoch - start_epoch + 1} epochs in {(time.time() - t0) / 3600:.3f} hours.")

        # Summary of saved results
        LOGGER.info(f"\n📁 Results Summary:")
        LOGGER.info(f"📊 All results saved to: {colorstr('bold', save_dir)}")
        LOGGER.info(f"🏆 Best model: {colorstr('bold', best)}")
        LOGGER.info(f"📈 Latest model: {colorstr('bold', last)}")
        LOGGER.info(
            f"📋 Hyperparameters: {colorstr('bold', save_dir / 'hyp.yaml')}")
        LOGGER.info(
            f"⚙️  Training options: {colorstr('bold', save_dir / 'opt.yaml')}")
        LOGGER.info(f"📊 TensorBoard logs: {colorstr('bold', save_dir)}")
        LOGGER.info(
            f"🌐 View TensorBoard: tensorboard --logdir {save_dir.parent}")
        LOGGER.info(f"🔗 Open: http://localhost:6006/")

        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\n🔍 Validating best model: {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(
                            mloss) + list(results) + lr, epoch, best_fitness, fi)

        # Final logging to TensorBoard
        if tb_writer:
            tb_writer.add_text("Training/Summary",
                               f"✅ Training completed: {epoch - start_epoch + 1} epochs, "
                               f"🏆 Best mAP@0.5: {best_fitness:.4f}, "
                               f"🎯 Final Precision: {results[0]:.4f}, "
                               f"📊 Final Recall: {results[1]:.4f}")

            # Create final results plot
            if plots:
                plot_results(save_dir / "results.csv")

        callbacks.run("on_train_end", last, best, epoch, results)

    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()
        LOGGER.info("✅ TensorBoard logging completed")
        LOGGER.info(f"\n🎯 Final Summary:")
        LOGGER.info(f"📁 All files saved to: {colorstr('bold', save_dir)}")
        LOGGER.info(f"🏆 Best model: {colorstr('bold', best)}")
        LOGGER.info(f"📊 TensorBoard: tensorboard --logdir {save_dir.parent}")
        LOGGER.info(f"🌐 View at: http://localhost:6006/")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
    Parse command-line arguments for YOLOv5 training with TensorBoard visualization.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.

    Links:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT /
                        "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT /
                        "dataset_allBB/data.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT /
                        "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=50,
                        help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int,
                        default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true",
                        help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True,
                        default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true",
                        help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true",
                        help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true",
                        help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true",
                        help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300,
                        help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None,
                        help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?",
                        const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true",
                        help="use weighted image selection for training")
    parser.add_argument("--device", default="",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true",
                        help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true",
                        help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str,
                        choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true",
                        help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8,
                        help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT /
                        "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true",
                        help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float,
                        default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100,
                        help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int,
                        default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1,
                        help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True,
                        default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1,
                        help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str,
                        default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console",
                        action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true",
                        help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """
    Enhanced main entry point for training with TensorBoard visualization.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().

    Returns:
        None
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(
            opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch_load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(
            opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            # if default project name, rename to runs/evolve
            if opt.project == str(ROOT / "runs/train"):
                opt.project = str(ROOT / "runs/evolve")
            # pass resume to exist_ok and disable resume
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != - \
            1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train with TensorBoard
    if not opt.evolve:
        train_with_tensorboard(opt.hyp, opt, device, callbacks)
    else:
        # Hyperparameter evolution (reuse original logic)
        LOGGER.warning(
            "Hyperparameter evolution not implemented with TensorBoard logging in this version")
        return


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
