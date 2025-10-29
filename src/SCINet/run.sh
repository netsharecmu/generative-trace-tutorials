python run_network.py \
    --data ugr16 \
    --root_path  /ocean/projects/cis230033p/yyin4/NetGPT/data/conext2023/small-scale/ugr16  \
    --data_path raw_bits_transformed_forecasting_10k.csv \
    --features M  \
    --seq_len 48 \
    --label_len 24 \
    --pred_len 24 \
    --hidden-size 4 \
    --stacks 1 \
    --levels 3 \
    --lr 3e-3 \
    --batch_size 64 \
    --train_epochs 1 \
    --dropout 0.5 \
    --inverse True \
    --skip_testing True \
    --data_split '0.7,0.3,0.0' \
    --freq 's'

# python run_network.py \
#     --data caida \
#     --root_path  /ocean/projects/cis230033p/yyin4/NetGPT/data/conext2023/small-scale/caida  \
#     --data_path raw_bits_transformed_forecasting_10k.csv \
#     --features M  \
#     --seq_len 48 \
#     --label_len 24 \
#     --pred_len 24 \
#     --hidden-size 4 \
#     --stacks 1 \
#     --levels 3 \
#     --lr 3e-3 \
#     --batch_size 64 \
#     --train_epochs 1 \
#     --dropout 0.5 \
#     --inverse True \
#     --skip_testing True \
#     --data_split '0.7,0.3,0.0' \
#     --freq 's'

# python run_network.py \
#     --data ugr16 \
#     --root_path  /ocean/projects/cis230033p/yyin4/NetGPT/data/conext2023/small-scale/ugr16  \
#     --data_path raw_bits_transformed_forecasting_10k.csv \
#     --features M  \
#     --seq_len 48 \
#     --label_len 24 \
#     --pred_len 24 \
#     --hidden-size 4 \
#     --stacks 1 \
#     --levels 3 \
#     --lr 3e-3 \
#     --batch_size 64 \
#     --train_epochs 1 \
#     --dropout 0.5 \
#     --generate True \
#     --inverse True \
#     --save True \
#     --sample_size 1000 \
#     --skip_testing True \
#     --data_split '0.7,0.3,0.0' \
#     --freq 's'