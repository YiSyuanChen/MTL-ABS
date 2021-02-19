################
##  Training  ##
################

##### Transfer Learning #####
#python train.py  -task abs -mode train \
# -bert_data_path ../datasets/bert_datasets/shard_10/aeslc \
# -model_path ../models/try/train \
# -log_path ../logs/try/train \
# -visible_gpus 0 \
# -save_checkpoint_steps 2 \
# -accum_count 1 \
# -batch_size 4 \
# -deterministic_batch_size true \
# -sep_optim true -use_bert_emb true -use_interval true \
# -lr_bert 0.0002 -lr_dec 0.0002 \
# -train_steps 2 -report_every 2 \
# -outer_no_warm_up true \
# -enc_adapter true \
# -dec_adapter true \
# -adapter_size 64 \
# -train_from ../models/pre_train/pre_ext_abs_cnndm/model_step_148000.pt \
# -ckpt_from_no_adapter true \
# -init_optim true \

#### Meta-Transfer Leanring ######
#python train.py  -task abs -mode train \
# -bert_data_path ../datasets/bert_meta_datasets/meta_data_rtw_40K \
# -model_path ../models/try/train \
# -log_path ../logs/try/train \
# -visible_gpus 0 \
# -save_checkpoint_steps 2 \
# -accum_count 1 \
# -batch_size 4 \
# -deterministic_batch_size true \
# -sep_optim true -use_bert_emb true -use_interval true \
# -lr_bert 0.0002 -lr_dec 0.0002 \
# -lr_bert_inner 0.0002 -lr_dec_inner 0.0002 \
# -train_steps 2 -report_every 1 \
# -inner_train_steps 4 -report_inner_every 4 \
# -outer_no_warm_up true \
# -inner_no_warm_up true \
# -dec_adapter true \
# -enc_adapter true \
# -adapter_size 64 \
# -meta_mode true \
# -num_batch_in_task 1 \
# -num_task 3 \
# -train_from ../models/pre_train/pre_ext_abs_cnndm/model_step_148000.pt \
# -ckpt_from_no_adapter true \
# -init_optim true \

##################
##  Validation  ##
##################

##### Transfer Learning #####
#python train.py -task abs -mode validate \
# -bert_data_path ../datasets/bert_datasets/shard_10/aeslc \
# -model_path ../models/try/train \
# -log_path ../logs/try/valid \
# -result_path ../results/try/valid \
# -visible_gpus 0 \
# -batch_size 3000 -test_batch_size 1500 \
# -sep_optim true -use_interval true \
# -max_pos 512 -max_length 15 -min_length 5 -alpha 0.7 \
# -enc_adapter true \
# -dec_adapter true \
# -adapter_size 64 \
# -test_all \

##### Meta-Transfer Leanring ######
#python train.py -task abs -mode validate \
# -bert_data_path ../datasets/bert_meta_datasets/meta_data_rtw_40K \
# -model_path ../models/try/train \
# -log_path ../logs/try/valid \
# -result_path ../results/try/valid \
# -visible_gpus 0 \
# -accum_count 1 \
# -batch_size 4 \
# -deterministic_batch_size true \
# -sep_optim true -use_bert_emb true -use_interval true \
# -lr_bert 0.0002 -lr_dec 0.0002 \
# -lr_bert_inner 0.0002 -lr_dec_inner 0.0002 \
# -train_steps 2 -report_every 1 \
# -inner_train_steps 8 -report_inner_every 8 \
# -outer_no_warm_up true \
# -inner_no_warm_up true \
# -dec_adapter true \
# -enc_adapter true \
# -adapter_size 64 \
# -meta_mode true \
# -num_batch_in_task 1 \
# -num_task 1 \
# -test_all \

###############
##  Testing  ##
###############

##### Meta-Transfer / Transfer Leanring ###### 
python train.py -task abs -mode test \
 -bert_data_path ../datasets/bert_datasets/shard_10/aeslc \
 -model_path ../models/try/train \
 -log_path ../logs/try/test \
 -result_path ../results/try/test \
 -visible_gpus 0 \
 -batch_size 3000 -test_batch_size 1500 \
 -sep_optim true -use_interval true \
 -max_pos 512 -max_length 15 -min_length 5 -alpha 0.7 \
 -enc_adapter true \
 -dec_adapter true \
 -adapter_size 64 \
 -test_from ../models/mtl_train/mtl_rtw/adaptation/10/aeslc/step_2700/train/model_step_14.pt \


