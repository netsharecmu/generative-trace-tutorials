# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

root_path_name=../../dataset/
data_path_name=ohe-netflow.csv
model_id_name=netflow
data_name=custom

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ../../dataset/ \
#   --data_path ohe-pcap.csv \
#   --model_id pcap_$seq_len'_'96 \
#   --model $model_name \
#   --data pcap \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 96 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log

# training 
python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name  \
  --model_id $model_id_name'_'$seq_len'_'192 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 17 \
  --des 'Exp' \
  --data_split '0.7,0.3,0.0' \
  --skip_testing True \
  --target 'td' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'192.log


# generation 
python -u run_longExp.py \
  --is_training 0 \
  --root_path $root_path_name \
  --data_path $data_path_name  \
  --model_id $model_id_name'_'$seq_len'_'192 \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 17 \
  --des 'Exp' \
  --target 'td' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'192.log

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'336.log

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_$seq_len'_'720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'720.log