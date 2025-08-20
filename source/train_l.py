import os
import torch
import datasets
import warnings
import numpy as np
import bitsandbytes as bnb

from torch import nn
from datasets import load_dataset
from dataclasses import dataclass, field
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from transformers.utils import logging as hf_logging

from eval import calc_result

datasets.disable_progress_bar()
hf_logging.set_verbosity_error()

warnings.filterwarnings(
    "ignore",
    message="The input hidden states seems to be silently casted in float32"
)

@dataclass
class Args(TrainingArguments):
    model_name: str = ""
    output_dir: str = ""
    data_dir: str = ""
    prompt_type: str = "both"

    packing: bool = False
    use_flash_attn: bool = True
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    warmup_ratio: float = 0.1

    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    max_seq_len: int = 1024
    weight_decay: float = 0.01

    lora_r: int = 8

    num_train_epochs: int = 10
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"

    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    report_to: str = "none"
    ddp_find_unused_parameters: bool = False
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    remove_unused_columns: bool = False

    optim: str = "paged_adamw_8bit"
    # disable_tqdm: bool = True

def find_all_linear_names(model: nn.Module) -> list[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        use_cache=False,
        num_labels=2,
    )

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
    )

    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r*2,
        target_modules=target_modules,
        bias="none",
        lora_dropout=0.1,
        task_type="SEQ_CLS",
    )

    peft_model: PeftModel = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    data_files = {
        split: os.path.join(args.data_dir, f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    train_datasets = load_dataset('json',data_files = data_files)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    def formatting_func(example):
        if args.prompt_type == "smiles":
            text = f"SMILES：{example['smiles']}\nLabel："
            return {"text": text}
        elif args.prompt_type == "description":
            text = f"Description：{example['description']}\nLabel："
            return {"text": text}
        elif args.prompt_type == "both":
            text = f"SMILES：{example['smiles']}\nDescription：{example['description']}\nLabel："
            return {"text": text}

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=False)

    formatted_datasets = train_datasets.map(
        formatting_func,
        remove_columns=['CID','smiles','description'],
        load_from_cache_file=False
    )

    tokenized_datasets = formatted_datasets.map(
        tokenize_function,
        batched=False,
        remove_columns=["text"],
        load_from_cache_file=False,
    )

    trainer = Trainer(
        args=args,
        model=peft_model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator = collate_fn,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    if trainer.state.best_model_checkpoint:
        model.load_adapter(trainer.state.best_model_checkpoint)

    predictions_dev = trainer.predict(tokenized_datasets["dev"])
    predictions_test = trainer.predict(tokenized_datasets["test"])

    def save_predictions(predictions, split_name):
        golds = predictions.label_ids
        logits = predictions.predictions
        pre, rec, f1, rocauc, prauc = calc_result(
            golds,
            logits.argmax(axis=1),
            torch.softmax(torch.tensor(logits), dim=1)[:,1]
        )
        with open(os.path.join(args.output_dir, f"result_{split_name}.txt"), "a") as f:
            f.write(f"Model: {args.data_dir}\n")
            f.write(f"Precision: {pre}\tRecall: {rec}\tF1 Score: {f1}\tROC AUC: {rocauc}\tPR AUC: {prauc}\n")

        logits_np = np.asarray(logits, dtype=np.float32)
        np.savetxt(os.path.join(args.output_dir, f"logits_{split_name}.txt"),logits_np,fmt="%.6f",delimiter="\t",header="\t".join([f"logit_{i}" for i in range(logits_np.shape[1])]),comments="")

    if local_rank == 0:
        save_predictions(predictions_dev, "dev")
        save_predictions(predictions_test, "test")

if __name__=="__main__":
    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    main(args)