GPU_ID=0
DISPLAY_ID=$((GPU_ID*5+7))
python ./train.py \
  --dataroot  ./datasets/a2b \
  --checkpoints_dir ./checkpoints \
  --exp_name dmit_a2b_transfer \
  --model_name a2b_transfer \
  --gpu ${GPU_ID} \
  --display_id ${DISPLAY_ID} \
  --display_port 8097 \
  --save_epoch_freq 25 \
  --niter 100 \
  --niter_decay 100 \
  --load_size 286 \
  --fine_size 256 \
  --n_attribute 2 \
  --n_style 8 \
  --batch_size 1 \
  --is_flip \
  --use_dropout \
  --print_freq 20









