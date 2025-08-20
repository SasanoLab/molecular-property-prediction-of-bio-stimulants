import os
import torch
import numpy as np

from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)

from eval import calc_result

@dataclass
class Args(TrainingArguments):
    model_name: str = ""
    peft_id: str = ""
    output_dir: str = ""
    data_dir: str = ""
    prompt_type: str = ""

def logit_to_probability(logit_1, logit_0):
    exp_1 = np.exp(logit_1)
    exp_0 = np.exp(logit_0)
    prob_1 = exp_1 / (exp_1 + exp_0)
    return prob_1

def main(args: Args):
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        attn_implementation="eager",
    )
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model=model, model_id=args.peft_id)

    def formatting_func(example):
        if args.prompt_type == "both":
            text = f"SMILES：{example['smiles']}\nDescription：{example['description']}\nLabel："
        else:
            text = f"SMILES：{example['smiles']}\nLabel："
        tokenized_input = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=1024
        )

        return {
            "input_ids": tokenized_input["input_ids"].squeeze(),
            "attention_mask": tokenized_input["attention_mask"].squeeze()
        }

    data_files = {}
    data_files["dev"] = os.path.join(args.data_dir,"dev.jsonl")
    data_files["test"] = os.path.join(args.data_dir,"test.jsonl")
    datasets = load_dataset('json',data_files = data_files)
    dev_dataset = datasets["dev"].map(formatting_func,batched = False,remove_columns=['CID','smiles','description'])
    test_dataset = datasets["test"].map(formatting_func,batched = False,remove_columns=['CID','smiles','description'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)

    dev_probs, dev_preds, dev_golds = [], [], []
    test_probs, test_preds, test_golds = [], [], []
    model.eval()

    id0 = tokenizer.encode("0", add_special_tokens=False)[0]
    id1 = tokenizer.encode("1", add_special_tokens=False)[0]

    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            inputs = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            dev_golds.append(batch['labels'].cpu())

            last_pos = attention_mask.sum().item() - 1
            last_token_logits = outputs.logits[:, last_pos, :]
            score0 = last_token_logits[0, id0].item()
            score1 = last_token_logits[0, id1].item()

            dev_probs.append(logit_to_probability(score1,score0))
            if score0 > score1:
                dev_preds.append(0)
            else:
                dev_preds.append(1)

        for batch in tqdm(test_dataloader):
            inputs = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            test_golds.append(batch['labels'].cpu())

            last_pos = attention_mask.sum().item() - 1
            last_token_logits = outputs.logits[:, last_pos, :]
            score0 = last_token_logits[0, id0].item()
            score1 = last_token_logits[0, id1].item()
            test_probs.append(logit_to_probability(score1,score0))
            if score0 > score1:
                test_preds.append(0)
            else:
                test_preds.append(1)

    def save_predictions(preds, probs, golds, split_name):
        pre, rec, f1, rocauc, prauc = calc_result(golds, preds, probs)
        with open(os.path.join(args.output_dir, f"result_{split_name}.txt"), "a") as f:
            f.write(f"Model: {args.data_dir}\n")
            f.write(f"Precision: {pre}\tRecall: {rec}\tF1 Score: {f1}\tROC AUC: {rocauc}\tPR AUC: {prauc}\n")
        with open(os.path.join(args.output_dir, f"logit_{split_name}.txt"), "a") as f:
            f.write("\n".join((probs)))

    save_predictions(dev_preds, dev_probs, dev_golds, "dev")
    save_predictions(test_preds, test_probs, test_golds, "test")


if __name__=="__main__":
    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    main(args)