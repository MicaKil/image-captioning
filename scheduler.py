class SchedulerWrapper:
    def __init__(self, scheduler, encoder_initial_lr):
        self.scheduler = scheduler
        self.encoder_initial_lr = encoder_initial_lr

    def step(self, metrics):
        self.scheduler.step(metrics)
        # Reset encoder's LR to initial value
        self.scheduler.optimizer.param_groups[0]['lr'] = self.encoder_initial_lr
