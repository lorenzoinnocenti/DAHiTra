gpus=0
checkpoint_root=checkpoints 
data_name=xBDataset 
dataset=xBDatasetMulti
loss=focal
n_class=5
lr=0.0002
lr_policy=multistep

img_size=512
batch_size=2

max_epochs=20
net_G=smaller

split=train  
split_val=val  
project_name=CROP_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}_ce_smoothen

python -u -B -O main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class}
# python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class}

# Different models trained:
# changeFormerV6
# unet_coupled_trans_256
# base_transformer_pos_s4_dd8
# base_transformer_pos_s4_dd8_o5
# newUNetTrans

# For xBD dataset:
# data_name=xBDataset 
# dataset=xBDatasetMulti
# loss=focal
# n_class=5
# lr=0.0002
# lr_policy=multistep
