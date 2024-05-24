# Train

screen -S train

conda deactivate; conda activate acrlsd

singularity shell --nv -B /data/base:/mnt --pwd /mnt/3M-APP-SCN/02_train/lsd /data/lsd_nm_experiments/lsd:v0.8.img

python train.py 300000