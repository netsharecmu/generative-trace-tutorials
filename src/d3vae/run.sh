# Install CUDNN version 7.6 first
# conda install -c conda-forge cudnn=7.6
# export LD_LIBRARY_PATH=/ocean/projects/cis230033p/yyin4/.conda/envs/d3vae/lib

# python main.py \
#     --data ugr16 \
#     --root_path /ocean/projects/cis230033p/yyin4/NetGPT/data/conext2023/small-scale/ugr16 \
#     --data_path raw_bits_transformed_forecasting.csv \
#     --input_dim 202 \
#     --target_dim 202 \
#     --percentage 1.0 \
#     --diff_steps 1000 \
#     --features='M' \
#     --beta_end 0.1 \
#     --inverse True \
#     --train_epochs 1 \
#     --itr 1 \
#     --skip_testing True \
#     --data_split '0.7,0.3,0.0'

# python main.py \
#     --root_path /ocean/projects/cis230033p/yyin4/NetGPT/data/conext2023/small-scale/caida \
#     --data_path raw_bits_transformed_forecasting.csv \
#     --input_dim 203 \
#     --target_dim 203 \
#     --percentage 1.0 \
#     --diff_steps 1000 \
#     --features='M' \
#     --beta_end 0.1 \
#     --inverse True \
#     --train_epochs 1 \
#     --itr 1 \
#     --skip_testing True \
#     --data_split '0.7,0.3,0.0'

python main.py \
    --data ugr16 \
    --work_folder ugr16_test \
    --root_path /ocean/projects/cis230033p/yyin4/NetGPT/data/conext2023/small-scale/ugr16 \
    --data_path raw_bits_transformed_forecasting.csv \
    --sample_size 10000 \
    --input_dim 202 \
    --target_dim 202 \
    --percentage 1.0 \
    --diff_steps 1000 \
    --features='M' \
    --beta_end 0.1 \
    --inverse True \
    --generate True \
    --train_epochs 1 \
    --itr 1 \
    --skip_testing True \
    --data_split '0.7,0.3,0.0'