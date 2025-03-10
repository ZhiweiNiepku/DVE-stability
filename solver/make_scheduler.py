import torch
from .cosine_lr import CosineLRScheduler

def make_scheduler(optimizer, lr, training_loader, epochs, scheduler_name, warm_up=5):
    if scheduler_name == "OneCycleLR":
        print("using OneCycleLR")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(training_loader), epochs=epochs)
    elif scheduler_name == "ExponentialLR":
        print("using ExponentialLR")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1, verbose=False)
    elif scheduler_name == "CosineLR":
        print("using CosineLR")
        num_epochs = epochs
        lr_min = 0.002 * lr
        warmup_lr_init = 0.01 * lr

        warmup_t = warm_up
        noise_range = None

        scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )

    return scheduler