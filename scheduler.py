class SchedulerWrapper:
    """
    A wrapper class for a learning rate scheduler that ensures the encoder's learning rate
    is reset to its initial value after each step.

    Attributes:
        scheduler: The learning rate scheduler to wrap.
        encoder_initial_lr: The initial learning rate for the encoder.
    """

    def __init__(self, scheduler, encoder_initial_lr):
        """
        Initialize the SchedulerWrapper.

        :param scheduler: The learning rate scheduler to wrap.
        :param encoder_initial_lr: The initial learning rate for the encoder.
        """
        self.scheduler = scheduler
        self.encoder_initial_lr = encoder_initial_lr

    def step(self, metrics):
        """
        Perform a step of the scheduler and reset the encoder's learning rate.

        :param metrics: The metric value to monitor for the scheduler.
        """
        self.scheduler.step(metrics)
        # Reset encoder's LR to initial value
        self.scheduler.optimizer.param_groups[0]['lr'] = self.encoder_initial_lr
