import os
import atexit
import datasets
import torch
import torch.nn as nn
import numpy as np
import bitsandbytes as bnb
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataclasses import dataclass, field
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers.utils import logging as hf_logging

from eval import calc_result
from model import NoSaveTrainer, BinaryClassificationModel

datasets.disable_progress_bar()
hf_logging.set_verbosity_error()


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    atexit.register(lambda: dist.destroy_process_group() if dist.is_initialized() else None)

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
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_steps: int = 500

    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})

    report_to: str = "none"
    ddp_find_unused_parameters: bool = True
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    save_safetensors: bool = False
    remove_unused_columns: bool = False

    optim: str = "paged_adamw_8bit"

def find_all_linear_names(model: nn.Module) -> list[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    is_distributed = dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

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

    model: PeftModel = get_peft_model(model, lora_config)

    if hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model.config, 'd_model'):
        hidden_size = model.config.d_model
    else:
        hidden_size = model.get_input_embeddings().embedding_dim

    model = BinaryClassificationModel(model, hidden_size, num_labels=2)

    data_files = {
        split: os.path.join(args.data_dir, f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    train_datasets = load_dataset('json', data_files=data_files)
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
        remove_columns=['CID', 'smiles', 'description'],
        load_from_cache_file=False
    )

    tokenized_datasets = formatted_datasets.map(
        tokenize_function,
        batched=False,
        remove_columns=["text"],
        load_from_cache_file=False,
    )

    trainer = NoSaveTrainer(
        args=args,
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        data_collator=collate_fn,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    if is_distributed:
        dist.barrier()

    torch.cuda.empty_cache()

    def predict(dataset, batch_size=args.per_device_eval_batch_size):
        model.eval()
        all_predictions = []
        all_labels = []
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = {k: v.to(model.base_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                outputs = model(**batch)

                logits = outputs.logits.detach().cpu()
                labels = batch['labels'].detach().cpu()

                all_predictions.append(logits)
                all_labels.append(labels)

                torch.cuda.empty_cache()

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return all_predictions.numpy(), all_labels.numpy()

    dev_predictions, dev_labels = predict(tokenized_datasets["dev"])
    test_predictions, test_labels = predict(tokenized_datasets["test"])

    def save_predictions(predictions, labels, split_name):
        golds = labels
        logits = predictions
        pre, rec, f1, rocauc, prauc = calc_result(
            golds,
            logits.argmax(axis=1),
            torch.softmax(torch.tensor(logits), dim=1)[:, 1]
        )
        with open(os.path.join(args.output_dir, f"result_{split_name}.txt"), "a") as f:
            f.write(f"Model: {args.data_dir}\n")
            f.write(f"Precision: {pre}\tRecall: {rec}\tF1 Score: {f1}\tROC AUC: {rocauc}\tPR AUC: {prauc}\n")

        logits_np = np.asarray(logits, dtype=np.float32)
        np.savetxt(os.path.join(args.output_dir, f"logits_{split_name}.txt"),logits_np,fmt="%.6f",delimiter="\t",header="\t".join([f"logit_{i}" for i in range(logits_np.shape[1])]),comments="")

    if local_rank == 0:
        save_predictions(dev_predictions, dev_labels, "dev")
        save_predictions(test_predictions, test_labels, "test")

    if is_distributed:
        dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    init_distributed()
    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    main(args)