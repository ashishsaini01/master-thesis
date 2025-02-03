from fastai.vision.all import *
from neptune.new.integrations.fastai import NeptuneCallback
from neptune.new.types import File

class NeptuneCallbackCustom(NeptuneCallback):
    @property
    def _optimizer_states(self) -> Optional[dict]:
        if hasattr(self, "opt") and hasattr(self.opt, "state"):
            state = [self.opt.state[p] for p,*_ in self.opt.all_params()]
            
            if len(state) == 1:
                return dict(state[0])

            return {
                f"group_layer_{layer}": {hyper: value for hyper, value in opts.items()}
                for layer, opts in enumerate(state)
            }

    def _log_learner_training_loop(self):
        s = io.StringIO()
        with redirect_stdout(s): self.learn.show_training_loop()

        self.neptune_run[f"{self.base_namespace}/config/learner"] = s.getvalue()

    def _log_ensemble_size(self):
        if hasattr(self.learn, 'max_samples'):
            self.neptune_run[f"{self.base_namespace}/config/ensemble_size"] = self.learn.max_samples

    def _log_attr_loss_func(self):
        if hasattr(self.learn, 'attr_loss_func'):
            attr_name = f"_{self.learn.attr_loss_func}-{self.learn.attr_reduction}"
            self.neptune_run[f"{self.base_namespace}/config/criterion"] = self._optimizer_criterion + attr_name

    def _log_model_configuration(self):
        super(NeptuneCallbackCustom, self)._log_model_configuration()

        self._log_learner_training_loop()
        self._log_ensemble_size()
        self._log_attr_loss_func()

    def _filter_states_epoch(self, states, prefix, step):
        try:
            if states.get('burn_in', 0):
                self.neptune_run[f"{prefix}/V_hat_inv_sqrt_min"].log(states["V_hat_inv_sqrt_min"].min().item(), step=step)
        except KeyError:
            pass

    def _filter_states_batch(self, states, prefix, step):
        try:
            if states.get('resample_prior', 0):
                self.neptune_run[f"{prefix}/weight_decay"].log(states["weight_decay"], step=step)
        except KeyError:
            pass

        try:
            if states.get('resample_momentum', 0):
                self.neptune_run[f"{prefix}/v_momentum_L2"].log(torch.linalg.norm(states["v_momentum"]), step)
        except KeyError:
            pass

    def _log_optimizer_states(self, prefix, func, **kwargs):
        if torch.tensor(["group_layer" in key for key in self._optimizer_states.keys()]).all():
            for param, value in self._optimizer_states.items():
                metric = f"{prefix}/{param}"
                func(value, metric, **kwargs)
        else:
                func(self._optimizer_states, prefix, **kwargs)
        
    def after_batch(self):
        prefix = f"{self.base_namespace}/metrics/fit_{self.fit_index}/{self._target}/batch"

        self.neptune_run[f"{prefix}/loss"].log(value=self.learn.loss.clone())
        if self._target != "validation":
            self.neptune_run[f"{prefix}/estimated_loss"].log(value=self.learn.loss_grad.clone())
            self._log_optimizer_states(f"{prefix}/optimizer_states", self._filter_states_batch, step=self.learn.train_iter)

        if hasattr(self, "smooth_loss"):
            self.neptune_run[f"{prefix}/smooth_loss"].log(value=self.learn.smooth_loss.clone())


    def after_train(self):
        super(NeptuneCallbackCustom, self).after_train()

        prefix = f"{self.base_namespace}/metrics/fit_{self.fit_index}/optimizer_states"
        self._log_optimizer_states(prefix, self._filter_states_epoch, step=self.learn.epoch)
