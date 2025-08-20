# !/bin/sh

MODEL="meta-llama/Llama-3.1-8B"
MODEL_NAME="llama"
PROMPT_TYPE="smiles"

for i in 1 2 3 4 5; do
    output_dir=../../output/${MODEL_NAME}/linear/${PROMPT_TYPE}/fold${i}
    FILE="$output_dir/result_test.txt"
    mkdir -p $output_dir
    uv run accelerate launch --config_file accelerate.json train_l.py \
        --model_name $MODEL \
        --output_dir $output_dir \
        --data_dir ../../data/pubchemstm/cleen/fold/set$i \
        --prompt_type $PROMPT_TYPE
done

for i in 1 2 3 4 5; do
    output_dir=../../output/debug/${MODEL_NAME}/gene/${PROMPT_TYPE}/fold${i}
    FILE="$output_dir/result_test.txt"
    mkdir -p $output_dir
    uv run accelerate launch --config_file accelerate.json train_g.py \
        --model_name $MODEL \
        --output_dir $output_dir \
        --data_dir ../../data/pubchemstm/cleen/fold/set$i \
        --prompt_type ${PROMPT_TYPE}
    uv run inf_g.py \
        --model_name $MODEL \
        --peft_id $output_dir \
        --output_dir $output_dir \
        --data_dir ../../data/pubchemstm/cleen/fold/set$i \
        --prompt_type ${PROMPT_TYPE}
done