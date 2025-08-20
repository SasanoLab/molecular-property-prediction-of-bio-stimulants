import torch
import torch.nn as nn
import functools

from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from torch.utils.checkpoint import checkpoint

class NoSaveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_state = None
        self.best_metric_value = None

    def _save_checkpoint(self, model, trial, metrics=None):
        if metrics is not None and self.args.metric_for_best_model in metrics:
            current_metric = metrics[self.args.metric_for_best_model]

            if (self.best_metric_value is None or
                (self.args.greater_is_better and current_metric > self.best_metric_value) or
                (not self.args.greater_is_better and current_metric < self.best_metric_value)):

                self.best_metric_value = current_metric

                import copy
                self.best_model_state = copy.deepcopy(model.state_dict())

    def _save(self, output_dir: str, state_dict=None):
        pass

    def save_model(self, output_dir=None, _internal_call=False):
        pass

    def load_best_model_in_memory(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            return True
        else:
            return False

class BinaryClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels=2):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self.gradient_checkpointing = False
        self.config = base_model.config
        self.config.num_labels = num_labels

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
        elif hasattr(self.base_model, 'enable_gradient_checkpointing'):
            self.base_model.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        if getattr(self, "_hf_peft_config_loaded", False):
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=checkpoint):
        is_gradient_checkpointing_set = False

        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]

        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            final_token_embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            final_token_embeddings = hidden_states[:, -1, :]
            
        pooled_output = self.dropout(final_token_embeddings)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, 'attentions', None)
        )