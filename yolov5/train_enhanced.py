# Enhanced YOLOv5 Training with Performance Optimization Techniques
"""
Enhanced YOLOv5 Training Script with Performance Optimization
- Data Augmentation Comparison
- Model Pruning Implementation
- Advanced Learning Rate Decay Functions
- Comprehensive TensorBoard Logging for Comparison

Usage:
    python train_enhanced.py --experiment baseline --model yolov5s
    python train_enhanced.py --experiment augmented --model yolov5s
    python train_enhanced.py --experiment pruned --model yolov5s
    python train_enhanced.py --experiment lr_decay --model yolov5s
    python train_enhanced.py --experiment combined --model yolov5s
    
    # For segmentation models
    python train_enhanced.py --experiment baseline --model yolov5s-seg
    python train_enhanced.py --experiment augmented --model yolov5s-seg
"""

from utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, select_device, smart_optimizer
)
from utils.metrics import fitness
from utils.loss import ComputeLoss
from utils.general import (
    LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_img_size,
    check_suffix, colorstr, increment_path, init_seeds, intersect_dicts,
    labels_to_class_weights, one_cycle, strip_optimizer, yaml_save
)
from utils.downloads import attempt_download
from utils.dataloaders import create_dataloader
from utils.callbacks import Callbacks
from utils.autobatch import check_train_batch_size
from utils.autoanchor import check_anchors
from models.experimental import attempt_load
from models.yolo import Model
import val as validate
from ultralytics.utils.patches import torch_load
import argparse
import math
import os
import random
import sys
import time
import copy
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class AdvancedDataAugmentation:
    """Enhanced data augmentation techniques for improved model convergence"""

    @staticmethod
    def get_enhanced_hyp():
        """Get enhanced hyperparameters for data augmentation"""
        return {
            # Enhanced augmentation settings
            'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,    # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,    # image HSV-Value augmentation (fraction)
            'degrees': 0.0,  # image rotation (+/- deg)
            'translate': 0.1,  # image translation (+/- fraction)
            'scale': 0.5,    # image scale (+/- gain)
            'shear': 0.0,    # image shear (+/- deg)
            # image perspective (+/- fraction), range 0-0.001
            'perspective': 0.0,
            'flipud': 0.0,   # image flip up-down (probability)
            'fliplr': 0.5,   # image flip left-right (probability)
            'mosaic': 1.0,   # image mosaic (probability)
            'mixup': 0.15,   # image mixup (probability)
            'copy_paste': 0.3,  # segment copy-paste (probability)

            # Advanced augmentation
            'erasing': 0.4,  # random erasing
            'crop': 0.7,     # random crop
        }


class ModelPruning:
    """Model pruning implementation for efficiency"""

    @staticmethod
    def prune_model(model, pruning_ratio=0.3):
        """Apply structured pruning to the model"""
        pruned_model = copy.deepcopy(model)

        # Apply pruning to convolutional layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Apply structured pruning (filter-wise)
                prune.ln_structured(module, name='weight',
                                    amount=pruning_ratio, n=2, dim=0)
                # Remove the pruning reparameterization
                prune.remove(module, 'weight')

        LOGGER.info(
            f"Applied {pruning_ratio*100}% pruning to convolutional layers")
        return pruned_model

    @staticmethod
    def calculate_sparsity(model):
        """Calculate model sparsity after pruning"""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params
        return sparsity


class AdvancedLRScheduler:
    """Advanced learning rate schedulers for better convergence"""

    @staticmethod
    def cosine_annealing_warm_restarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6):
        """Cosine Annealing with Warm Restarts"""
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)

    @staticmethod
    def exponential_decay(optimizer, gamma=0.95):
        """Exponential decay scheduler"""
        return lr_scheduler.ExponentialLR(optimizer, gamma)

    @staticmethod
    def step_decay(optimizer, step_size=30, gamma=0.1):
        """Step decay scheduler"""
        return lr_scheduler.StepLR(optimizer, step_size, gamma)

    @staticmethod
    def polynomial_decay(optimizer, max_epochs, power=0.9):
        """Polynomial decay scheduler"""
        def poly_lr_lambda(epoch):
            return (1 - epoch / max_epochs) ** power
        return lr_scheduler.LambdaLR(optimizer, poly_lr_lambda)


def train_enhanced(hyp, opt, device, callbacks, experiment_type):
    """Enhanced training function with optimization techniques"""

    save_dir = Path(opt.save_dir)
    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights

    # Create experiment-specific directory
    exp_dir = save_dir / f"{experiment_type}_{opt.model_type}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard with experiment name
    tb_writer = SummaryWriter(str(exp_dir / "tensorboard"))

    LOGGER.info(
        f"🚀 Starting {experiment_type} experiment with {opt.model_type}")
    LOGGER.info(f"📊 Results will be saved to: {colorstr('bold', exp_dir)}")
    LOGGER.info(f"📈 TensorBoard: {colorstr('bold', exp_dir / 'tensorboard')}")

    # Load and modify hyperparameters based on experiment
    if experiment_type == "augmented":
        enhanced_hyp = AdvancedDataAugmentation.get_enhanced_hyp()
        hyp.update(enhanced_hyp)
        LOGGER.info("🔄 Enhanced data augmentation applied")

    # Dataset
    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = int(data_dict["nc"])
    names = data_dict["names"]

    # Model
    check_suffix(weights, ".pt")
    pretrained = weights.endswith(".pt")

    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch_load(weights, map_location="cpu")
        model = Model(ckpt["model"].yaml, ch=3, nc=nc,
                      anchors=hyp.get("anchors")).to(device)
        exclude = ["anchor"]
        csd = ckpt["model"].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(
            f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    else:
        model = Model(opt.cfg, ch=3, nc=nc,
                      anchors=hyp.get("anchors")).to(device)

    # Apply model pruning if specified
    original_model = copy.deepcopy(model)
    if experiment_type == "pruned" or experiment_type == "combined":
        model = ModelPruning.prune_model(model, pruning_ratio=0.3)
        sparsity = ModelPruning.calculate_sparsity(model)
        LOGGER.info(f"✂️ Model pruned with sparsity: {sparsity:.2%}")

        # Log model size comparison
        original_size = sum(p.numel() for p in original_model.parameters())
        pruned_size = sum(p.numel()
                          for p in model.parameters() if p.requires_grad)
        LOGGER.info(
            f"📊 Model size reduction: {original_size} → {pruned_size} ({(1-pruned_size/original_size)*100:.1f}% reduction)")

    amp = check_amp(model)

    # Image size and batch size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp["weight_decay"] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(
        model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Advanced Learning Rate Scheduler
    if experiment_type == "lr_decay" or experiment_type == "combined":
        if opt.lr_scheduler == "cosine_restart":
            scheduler = AdvancedLRScheduler.cosine_annealing_warm_restarts(
                optimizer, T_0=10, T_mult=2)
            LOGGER.info("📈 Using Cosine Annealing with Warm Restarts")
        elif opt.lr_scheduler == "exponential":
            scheduler = AdvancedLRScheduler.exponential_decay(
                optimizer, gamma=0.95)
            LOGGER.info("📈 Using Exponential Decay")
        elif opt.lr_scheduler == "polynomial":
            scheduler = AdvancedLRScheduler.polynomial_decay(
                optimizer, epochs, power=0.9)
            LOGGER.info("📈 Using Polynomial Decay")
        else:
            scheduler = AdvancedLRScheduler.step_decay(
                optimizer, step_size=epochs//3, gamma=0.1)
            LOGGER.info("📈 Using Step Decay")
    else:
        # Standard scheduler
        if opt.cos_lr:
            lf = one_cycle(1, hyp["lrf"], epochs)
        else:
            def lf(x): return (1 - x / epochs) * \
                (1.0 - hyp["lrf"]) + hyp["lrf"]
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        LOGGER.info("📈 Using Standard Learning Rate Schedule")

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Data loaders
    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, False,
        hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect,
        rank=LOCAL_RANK, workers=opt.workers, image_weights=opt.image_weights,
        quad=opt.quad, prefix=colorstr("train: "), shuffle=True, seed=opt.seed,
    )

    val_loader = create_dataloader(
        val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, False,
        hyp=hyp, cache=opt.cache, rect=True, rank=-1,
        workers=opt.workers * 2, pad=0.5, prefix=colorstr("val: "),
    )[0]

    # Model attributes
    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl
    hyp["cls"] *= nc / 80 * 3 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc
    model.names = names

    # Training setup
    nb = len(train_loader)
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = -1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)

    # Log initial metrics
    tb_writer.add_text("Experiment/Type", experiment_type)
    tb_writer.add_text("Experiment/Model", opt.model_type)
    tb_writer.add_scalar("Config/Epochs", epochs, 0)
    tb_writer.add_scalar("Config/BatchSize", batch_size, 0)
    tb_writer.add_scalar("Config/LearningRate", hyp["lr0"], 0)

    # Training loop
    LOGGER.info(f"🏃‍♂️ Starting training for {epochs} epochs...")
    t0 = time.time()
    best_fitness = 0.0

    for epoch in range(epochs):
        model.train()

        mloss = torch.zeros(3, device=device)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem",
                    "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"]])
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Forward pass
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    loss *= WORLD_SIZE

            # Backward pass
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Logging
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(("%11s" * 2 + "%11.4g" * 5) %
                                     (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1]))

                # TensorBoard logging (batch level)
                if ni % 100 == 0:
                    tb_writer.add_scalar("Loss/Train_Box", mloss[0], ni)
                    tb_writer.add_scalar("Loss/Train_Objectness", mloss[1], ni)
                    tb_writer.add_scalar(
                        "Loss/Train_Classification", mloss[2], ni)
                    tb_writer.add_scalar("Loss/Train_Total", mloss.sum(), ni)
                    tb_writer.add_scalar(
                        "Learning_Rate", optimizer.param_groups[0]["lr"], ni)

        # Scheduler step
        lr = [x["lr"] for x in optimizer.param_groups]
        scheduler.step()

        # Validation
        if RANK in {-1, 0}:
            ema.update_attr(
                model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            if not opt.noval or final_epoch:
                results, maps, _ = validate.run(
                    data_dict, batch_size=batch_size // WORLD_SIZE * 2, imgsz=imgsz,
                    half=amp, model=ema.ema, single_cls=False, dataloader=val_loader,
                    save_dir=exp_dir, plots=False, callbacks=callbacks, compute_loss=compute_loss,
                )

            # TensorBoard logging (epoch level)
            tb_writer.add_scalar("Epoch/Train_Loss_Box", mloss[0], epoch)
            tb_writer.add_scalar(
                "Epoch/Train_Loss_Objectness", mloss[1], epoch)
            tb_writer.add_scalar(
                "Epoch/Train_Loss_Classification", mloss[2], epoch)
            tb_writer.add_scalar("Epoch/Train_Loss_Total", mloss.sum(), epoch)

            # Validation metrics
            tb_writer.add_scalar("Metrics/Precision", results[0], epoch)
            tb_writer.add_scalar("Metrics/Recall", results[1], epoch)
            tb_writer.add_scalar("Metrics/mAP_0.5", results[2], epoch)
            tb_writer.add_scalar("Metrics/mAP_0.5:0.95", results[3], epoch)

            # Per-class mAP
            for j, c in enumerate(names.values()):
                if j < len(maps):
                    tb_writer.add_scalar(f"mAP_per_class/{c}", maps[j], epoch)

            # Update best fitness
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
                # Save best model
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": de_parallel(model).half(),
                    "ema": ema.ema.half() if ema else None,
                    "optimizer": optimizer.state_dict(),
                    "date": datetime.now().isoformat(),
                }
                torch.save(ckpt, exp_dir / "weights" / "best.pt")

            # Early stopping
            if stopper(epoch=epoch, fitness=fi):
                break

    # Training completed
    training_time = (time.time() - t0) / 3600
    LOGGER.info(
        f"🎉 {experiment_type} training completed in {training_time:.2f} hours")
    LOGGER.info(f"🏆 Best mAP@0.5: {best_fitness:.4f}")
    LOGGER.info(f"� Results saved to: {exp_dir}")

    # Final summary to TensorBoard
    tb_writer.add_text("Training/Summary",
                       f"Experiment: {experiment_type}\n"
                       f"Model: {opt.model_type}\n"
                       f"Training time: {training_time:.2f}h\n"
                       f"Best mAP@0.5: {best_fitness:.4f}\n"
                       f"Final Precision: {results[0]:.4f}\n"
                       f"Final Recall: {results[1]:.4f}")

    # Log experiment results
    tb_writer.add_scalar("Final/Best_Fitness", best_fitness, 0)
    tb_writer.add_scalar("Final/Training_Time", training_time, 0)

    tb_writer.close()

    return {
        "experiment_type": experiment_type,
        "model_type": opt.model_type,
        "best_fitness": best_fitness,
        "final_results": results,
        "training_time": training_time,
        "save_dir": exp_dir
    }


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default="dataset_allBB/data.yaml", help="dataset.yaml path")
    parser.add_argument(
        "--hyp", type=str, default="data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=50,
                        help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="total batch size for all GPUs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true",
                        help="rectangular training")
    parser.add_argument("--resume", action="store_true",
                        help="resume most recent training")
    parser.add_argument("--nosave", action="store_true",
                        help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true",
                        help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true",
                        help="disable AutoAnchor")
    parser.add_argument("--cache", type=str, nargs="?",
                        const="ram", help="--cache images in ram/disk")
    parser.add_argument("--image-weights", action="store_true",
                        help="use weighted image selection for training")
    parser.add_argument("--device", default="",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true",
                        help="vary img-size +/- 50%")
    parser.add_argument("--single-cls", action="store_true",
                        help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str,
                        choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true",
                        help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8,
                        help="max dataloader workers")
    parser.add_argument("--project", default="runs/enhanced",
                        help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true",
                        help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float,
                        default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100,
                        help="EarlyStopping patience")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global training seed")

    # Enhanced training arguments
    parser.add_argument("--experiment", type=str, default="baseline",
                        choices=["baseline", "augmented",
                                 "pruned", "lr_decay", "combined"],
                        help="Experiment type")
    parser.add_argument("--model-type", type=str, default="yolov5s",
                        choices=["yolov5s", "yolov5s-seg"], help="Model type")
    parser.add_argument("--lr-scheduler", type=str, default="cosine_restart",
                        choices=["cosine_restart",
                                 "exponential", "polynomial", "step"],
                        help="Learning rate scheduler type")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all experiments")

    return parser.parse_args()


def main(opt):
    """Main function to run experiments"""

    # Setup
    init_seeds(opt.seed)
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Load hyperparameters
    with open(opt.hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)

    # Setup save directory
    save_dir = increment_path(
        Path(opt.project) / opt.name, exist_ok=opt.exist_ok)

    # Update weights path based on model type
    if opt.model_type == "yolov5s-seg":
        opt.weights = "yolov5s-seg.pt"
    else:
        opt.weights = "yolov5s.pt"

    opt.save_dir = str(save_dir)

    callbacks = Callbacks()

    results_summary = []

    if opt.run_all:
        # Run all experiments
        experiments = ["baseline", "augmented",
                       "pruned", "lr_decay", "combined"]
        LOGGER.info(f"� Running all experiments with {opt.model_type}")

        for exp in experiments:
            LOGGER.info(f"\n{'='*50}")
            LOGGER.info(f"🧪 Starting {exp} experiment")
            LOGGER.info(f"{'='*50}")

            opt.experiment = exp
            result = train_enhanced(hyp.copy(), opt, device, callbacks, exp)
            results_summary.append(result)

            # Clear GPU memory
            torch.cuda.empty_cache()
    else:
        # Run single experiment
        LOGGER.info(
            f"🧪 Running {opt.experiment} experiment with {opt.model_type}")
        result = train_enhanced(hyp.copy(), opt, device,
                                callbacks, opt.experiment)
        results_summary.append(result)

    # Print comparison results
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("📊 EXPERIMENT RESULTS COMPARISON")
    LOGGER.info(f"{'='*60}")

    for result in results_summary:
        LOGGER.info(f"🧪 {result['experiment_type']:12} | "
                    f"🏆 mAP@0.5: {result['best_fitness']:.4f} | "
                    f"⏱️  Time: {result['training_time']:.2f}h | "
                    f"📁 {result['save_dir']}")

    # Find best experiment
    if len(results_summary) > 1:
        best_exp = max(results_summary, key=lambda x: x['best_fitness'])
        LOGGER.info(f"\n🥇 Best experiment: {best_exp['experiment_type']} "
                    f"(mAP@0.5: {best_exp['best_fitness']:.4f})")

    LOGGER.info(f"\n📊 TensorBoard comparison: tensorboard --logdir {save_dir}")
    LOGGER.info(f"🌐 View at: http://localhost:6006/")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
