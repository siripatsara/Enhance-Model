# Learning Rate Decay Strategies for Lab-5
# เทคนิค Decay Learning Rate Function

import math
import torch.optim.lr_scheduler as lr_scheduler


class EnhancedLRSchedulers:
    """เก็บ LR schedulers ต่างๆ สำหรับการเปรียบเทียบ"""

    @staticmethod
    def cosine_annealing(optimizer, T_max, eta_min=1e-6):
        """Cosine Annealing with restart"""
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

    @staticmethod
    def cosine_warm_restart(optimizer, T_0=10, T_mult=2, eta_min=1e-6):
        """Cosine Annealing with Warm Restarts"""
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )

    @staticmethod
    def exponential_decay(optimizer, gamma=0.95):
        """Exponential Learning Rate Decay"""
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    @staticmethod
    def step_decay(optimizer, step_size=30, gamma=0.5):
        """Step Learning Rate Decay"""
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

    @staticmethod
    def multi_step_decay(optimizer, milestones=[50, 80], gamma=0.3):
        """Multi-step Learning Rate Decay"""
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )

    @staticmethod
    def plateau_decay(optimizer, patience=10, factor=0.5, min_lr=1e-7):
        """Reduce LR on Plateau"""
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )

    @staticmethod
    def polynomial_decay(optimizer, total_epochs, power=0.9):
        """Polynomial Learning Rate Decay"""
        def lambda_func(epoch): return (1 - epoch / total_epochs) ** power
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)


# Configuration สำหรับแต่ละ strategy
LR_STRATEGIES = {
    'cosine': {
        'scheduler': 'cosine_annealing',
        'params': {'T_max': 100, 'eta_min': 1e-6},
        'description': 'Cosine annealing - smooth decay'
    },
    'cosine_restart': {
        'scheduler': 'cosine_warm_restart',
        'params': {'T_0': 20, 'T_mult': 2},
        'description': 'Cosine with warm restarts'
    },
    'exponential': {
        'scheduler': 'exponential_decay',
        'params': {'gamma': 0.96},
        'description': 'Exponential decay'
    },
    'step': {
        'scheduler': 'step_decay',
        'params': {'step_size': 25, 'gamma': 0.5},
        'description': 'Step decay every 25 epochs'
    },
    'multi_step': {
        'scheduler': 'multi_step_decay',
        'params': {'milestones': [40, 70, 90], 'gamma': 0.3},
        'description': 'Multi-step decay at milestones'
    },
    'plateau': {
        'scheduler': 'plateau_decay',
        'params': {'patience': 8, 'factor': 0.6},
        'description': 'Reduce on plateau (adaptive)'
    },
    'polynomial': {
        'scheduler': 'polynomial_decay',
        'params': {'total_epochs': 100, 'power': 0.9},
        'description': 'Polynomial decay'
    }
}
