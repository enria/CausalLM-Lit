import shutil
import os
import lightning.pytorch as pl

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from weakref import proxy

class RenameBestCheckpointCallback(Callback):
    def __init__(self, ckpt_callback:ModelCheckpoint, best_name="best.ckpt") -> None:
        super().__init__()
        self.ckpt_callback = ckpt_callback
        self.best_name=best_name

    def on_fit_end(self, trainer, pl_module) -> None:
        renamed_best_path = os.path.join(self.ckpt_callback.dirpath, self.best_name)
        rank_zero_info(
            f"Finished fit stage, copying and rename the best ckpt to: {renamed_best_path}"
        )
        shutil.copy(self.ckpt_callback.best_model_path, renamed_best_path)

from lightning.pytorch.callbacks import ModelCheckpoint


class PeftCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        trainer.strategy._lightning_module.model.save_pretrained(filepath)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

        
