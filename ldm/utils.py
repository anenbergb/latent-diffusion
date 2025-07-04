import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup_then_constant(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_constant_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule where the learning rate:
      1. Linearly increases from 0 to the optimizer's initial LR over `num_warmup_steps`.
      2. Stays constant at the max LR for `num_constant_steps`.
      3. Then decreases following a cosine curve to 0 for the remaining steps.

    Args:
        optimizer (`torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            Number of steps for the warmup phase.
        num_constant_steps (`int`):
            Number of steps to hold the learning rate at max after warmup.
        num_training_steps (`int`):
            Total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            Number of cosine cycles (default 0.5 = one half-cosine from max to 0).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the specified schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        elif current_step < num_warmup_steps + num_constant_steps:
            # Constant max LR
            return 1.0

        else:
            # Cosine decay
            decay_start = num_warmup_steps + num_constant_steps
            decay_steps = num_training_steps - decay_start
            if decay_steps <= 0:
                return 0.0  # training ends before cosine starts
            progress = float(current_step - decay_start) / float(decay_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
