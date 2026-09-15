from lib.logger import print_

def update_scheduler(scheduler, exp_params, control_metric=None, iter=-1, end_epoch=False, section="training_slots"):
    scheduler_type = exp_params[section]["scheduler"]
    if scheduler is None:
        return
    if(scheduler_type == "plateau" and end_epoch):
        scheduler.step(control_metric)
    elif(scheduler_type in ["step", "multi_step"] and end_epoch):
        scheduler.step()
    elif(scheduler_type == "exponential" and not end_epoch):
        scheduler.step(iter)
    elif(scheduler_type == "cosine_annealing" and end_epoch):
        scheduler.step()
    else:
        pass
    return

class ExponentialLRSchedule:
    def __init__(self, optimizer, init_lr, gamma=0.5, total_steps=1_000_000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.total_steps = total_steps
        return

    def update_lr(self, step):
        new_lr = self.init_lr * self.gamma ** (step / self.total_steps)
        return new_lr

    def step(self, iter):
        if(iter < self.total_steps):
            for params in self.optimizer.param_groups:
                params["lr"] = self.update_lr(iter)
        elif(iter == self.total_steps):
            print_(f"Finished exponential decay due to reach of {self.total_steps} steps")
        return

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        self.init_lr = state_dict["init_lr"]
        self.gamma = state_dict["gamma"]
        self.total_steps = state_dict["total_steps"]
        return

class LRWarmUp:
    def __init__(self, init_lr, warmup_steps, max_epochs=1):
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.active = True
        self.final_step = -1

    def __call__(self, iter, epoch, optimizer):
        if(iter > self.warmup_steps):
            if(self.active):
                self.final_step = iter
                self.active = False
                lr = self.init_lr
                print_("Finished learning rate warmup period...")
                print_(f"  --> Reached iter {iter} >= {self.warmup_steps}")
                print_(f"  --> Reached at epoch {epoch}")
        elif(epoch >= self.max_epochs):
            if(self.active):
                self.final_step = iter
                self.active = False
                lr = self.init_lr
                print_("Finished learning rate warmup period:")
                print_(f"  --> Reached epoch {epoch} >= {self.max_epochs}")
                print_(f"  --> Reached at iter {iter}")
        else:
            if iter >= 0:
                lr = self.init_lr * (iter / self.warmup_steps)
                for params in optimizer.param_groups:
                    params["lr"] = lr
        return

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict if isinstance(state_dict, dict) else state_dict.state_dict())
        return

class WarmupVSScehdule:
    def __init__(self, optimizer, lr_warmup, scheduler, section="training_slots"):
        self.optimizer = optimizer
        self.lr_warmup = lr_warmup
        self.scheduler = scheduler
        self.section = section
        return

    def __call__(self, iter, epoch, exp_params, end_epoch, control_metric=None):
        if self.lr_warmup.active:
            self.lr_warmup(iter=iter, epoch=epoch, optimizer=self.optimizer)
        else:
            update_scheduler(
                    scheduler=self.scheduler,
                    exp_params=exp_params,
                    iter=iter - self.lr_warmup.final_step - 1,
                    end_epoch=end_epoch,
                    control_metric=control_metric,
                    section=self.section
            )
        return
