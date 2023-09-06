
from peft import set_peft_model_state_dict, get_peft_model_state_dict

class PeftModuleMixin:
    def state_dict(self):
        if not self.is_peft_mode:
            return super().state_dict()
        else:
            return get_peft_model_state_dict( self.model )

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self.is_peft_mode:
            return super().load_state_dict(state_dict, strict)
        else:
            set_peft_model_state_dict(self.model, state_dict)

class PeftModuleMixin:
    def state_dict(self):
        if not self.is_peft_mode:
            return super().state_dict()
        else:
            return get_peft_model_state_dict( self.model )

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self.is_peft_mode:
            return super().load_state_dict(state_dict, strict)
        else:
            set_peft_model_state_dict(self.model, state_dict)