# Train the base model on ShanghaiTech A
python trainer.py \
    --dataset sha --input_size 224 --block_size 16 \
    --model_name ebc_b --num_vpt 96 --sliding_window --warmup_lr 1e-3 \
    --amp --num_workers 8

# # Train the base model on ShanghaiTech B. You can also try block_size 32.
# python trainer.py \
#     --dataset sha --input_size 448 --block_size 16 \
#     --model_name ebc_b --amp --num_workers 8

# # Train the base model on UCF-QRNF.
# python trainer.py \
#     --dataset qnrf --input_size 672 --block_size 32 \
#     --model_name ebc_b --amp --num_workers 8

# # Train the base model on NWPU-Crowd. You can also try block_size 16 or 32.
# python trainer.py \
#     --dataset sha --input_size 672 --block_size 8 \
#     --model_name ebc_b --amp --num_workers 8
