# Meta-Transfer Learning for Low-Resource Abstractive Summarization (MTL-ABS)
![](https://img.shields.io/badge/license-MIT-brightgreen) ![](https://img.shields.io/badge/Python-3.6-blue) ![](https://img.shields.io/badge/Pytorch-1.1.0-orange)

<p align="center">
  <img src="https://github.com/YiSyuanChen/MTL-ABS/blob/main/framework.png" width="372" height="368">
</p>

**[IMPORTANT] We are currently re-organizing and cleaning code for better usage, and we will keep update this repository recently.**

# Introduction

Original PyTorch implementation for AAAI 2021 Paper "Meta-Transfer Learning for Low-Resrouce Abstractive Summarization" by Yi-Syuan Chen and Hong-Han Shuai.

If you have any questions on this repository or the related paper, feel free to create an issue or send me an email.


# File Structure

# Performance

| Dataset       | `MTL-ABS`       | `PEGASUS`  |
| ------------- |:-------------:| :-----:|
| `ALSEC`       | 21.27/10.79/20.85     | 11.97/4.91/10.84 |
| `BillSum`       | 41.22/18.61/26.33     | 40.48/18.49/27.27 |
| `Gigaword`       | 28.98/11.86/26.74     | 25.32/8.88/22.55 |
| `Multi-News`       | 38.88/12.78/19.88     | 39.79/12.56/20.06 |
| `NEWSROOM`       | 37.15/25.40/33.78     | 29.24/17.78/24.98 |
| `Reddit-TIFU`       | 18.03/6.41/17.10     | 15.36/2.91/10.76 |
| `arXiv`       | 35.81/10.26/20.51     | 31.38/8.16/17.97 |
| `PubMed`       | 34.08/10.05/18.66     | 33.31/10.58/20.05 |
| `WikiHow`       | 28.34/8.16/19.72     | 23.95/6.54/15.33 |



# Instructions
## Dataset
Following steps are organized in the script _src/prepro_dataset.sh_.
### Step 1. Download Stanford CoreNLP
Download the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and add the following command to bash file:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
This tool will be used for preliminary tokenization (Step 3).

### Step 2. Download TFDS Dataset
Download and format the dataset from [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets).
```
python prepro/prepro_tfds_dataset.py -mode download -dataset aeslc -output_path ../datasets/organized_raw_datasets/ -art_feature_name email_body -summ_feature_name subject_line                                                                                  
```
It will produce raw text files containing source text and target text to folder _datasets/organized_raw_datasets_. The source and target is seperated with special token "@highlight".

The essential arguments include:
- mode : download dataset (download) or split sentence for exist dataset (ssplit).
- dataset: the name of dataset on TFDS.
- art_feature_name : the feature name for source text on TFDS.
- summ_feature_num : the feature name for target text on TFDS.

**NOTE**: 
- In some dataset, the source sentences are not speperated (e.g. CNN/DailyMail). One can split it with -ssplit option. 
- There is no default training/testing split for Reddit and Reddit-TIFU. One need manually split them.
- The BIGPATENT dataset is large and requires Apache Beam to download. We instead use the [raw dataset](https://evasharma.github.io/bigpatent/) and process it with _prepro_bigpatent_dataset.py_.


### Step 3. Tokenization
Use Stanford CoreNLP to perform preliminary tokenization. 
```
python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/aeslc/ -save_path ../datasets/tokenized_dataset/aeslc/ -limit_num_file_tokenized 40000                                                                                      
```
It will produce tokenized texts in a json format to folder _datasets/tokenized_dataset_.

The essential arguments include:
- limit_num_file_tokenized : number of files to be tokenized.

### Step 4. Extract Required Information
The resulted files in Step 3. contain many pieces of information. We only extract the raw tokens and aggregate the tokens from several data to create data shards. Each shard will be stored in one file.   
```
python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/aeslc/ -save_path ../datasets/json_dataset/aeslc/ -shard_size 10                                                                                                          
```
It will produce shards of data in a json format to folder _datasets/json_dataset_.

The essential arguments include:
- shard_size : number of data in a json file.

### Step 5. Preprocess for BERT Pipeline
Re-tokenize data with BERT tokenizer, add special tokens (e.g. [CLS], [SEP]), and create all require data for BERT pipeline. 
```
python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/aeslc/ -save_path ../datasets/bert_datasets/shard_10/aeslc/ -max_src_ntokens 512 -max_tgt_ntokens 32                                                                                 
```
It will produce shards of data in binary format for BERT pipeline.

The essential arguments include:
- min_src_sents : ignore data without enough source sentences.
- max_src_sents : truncate sentences in source.
- min_src_ntokens_per_sent : ignore source sentence without enough tokens.
- max_src_ntokens_per_sent : truncate tokens in source sentences. 
- max_src_ntokens : truncate tokens in source.
- min_tgt_ntokens : ignore data without enough target tokens.
- max_tgt_ntokens : truncate tokens in target.


### Step 6. Create Meta-Dataset
Create meta-dataset from datasets on hand (Step 2~5). 
```
python prepro/prepro_meta_dataset.py -input_path ../datasets/bert_datasets/shard_10 -output_path ../datasets/bert_meta_datasets -abbrev rtw_40K -max_train_pt_files 4000 -max_valid_pt_files 4000 -train_dataset_list reddit,reddit_tifu,wikihow -valid_dataset_list aeslc
python prepro/rename_train_to_valid.py -input_path ../datasets/bert_meta_datasets/meta_data_rtw_40K/valid
```
The resulted file structure:
```
meta_data_<abbrev>
  - train
    - support
      - <assigned_dataset_1>
      - <assigned_dataset_2>
      ...
    - query
  - valid
    - support
    - query
```

The essential arguments include:
- max_train_pt_files : number of binary files for each support set.
- max_valid_pt_files : number of binary files for each query set.
- train_dataset_list : dataset to be used in meta-training.
- valid_dataset_list : dataset to be used in meta-validation.

**NOTE**: 
- Since we use training data (with 'train' in file name) to create meta-dataset, for meta-validation, we need to modify file name with _rename_train_to_valid.py_.

## Training 
``` 
python train.py  -task abs -mode train \                                                                                                                                                                                                                         
  -bert_data_path ../datasets/bert_meta_datasets/meta_data_rtw_40K \                                                                                                                                                                                                    
  -model_path ../models/adapter_dev/train \                                                                                                                                                                                                                           
  -log_path ../logs/adapter_dev/train \                                                                                                                                                                                                                             
  -visible_gpus 0 \                                                                                                                                                                                                                                                 
  -save_checkpoint_steps 2 \                                                                                                                                                                                                                                        
  -accum_count 1 \                                                                                                                                                                                                                                                  
  -batch_size 4 \                                                                                                                                                                                                                                                   
  -deterministic_batch_size true \                                                                                                                                                                                                                                  
  -sep_optim true -use_bert_emb true -use_interval true \                                                                                                                                                                                                          
  -lr_bert 0.0002 -lr_dec 0.0002 \                                                                                                                                                                                                                                  
  -lr_bert_inner 0.0002 -lr_dec_inner 0.0002 \                                                                                                                                                                                                                      
  -train_steps 2 -report_every 1 \                                                                                                                                                                                                                                  
  -inner_train_steps 4 -report_inner_every 4 \                                                                                                                                                                                                                      
  -outer_no_warm_up true \                                                                                                                                                                                                                                          
  -inner_no_warm_up true \                                                                                                                                                                                                                                          
  -dec_adapter true \                                                                                                                                                                                                                                               
  -enc_adapter true \                                                                                                                                                                                                                                               
  -adapter_size 64 \                                                                                                                                                                                                                                                
  -meta_mode true \                                                                                                                                                                                                                                                
  -num_batch_in_task 1 \                                                                                                                                                                                                                                           
  -num_task 3 \                                                                                                                                                                                                                                                     
  -train_from ../models/pre_train/pre_ext_abs_cnndm/model_step_148000.pt \                                                                                                                                                                                          
  -ckpt_from_no_adapter true \                                                                                                                                                                                                                                      
  -init_optim true \                                              
```

## Validation
```
python train.py -task abs -mode validate \                                                                                                                                                                                                                                                                                                                                                                                                                                                            
  -bert_data_path ../datasets/bert_meta_datasets/meta_data_rtw_40K \                                                                                                                                                                                               
  -model_path ../models/adapter_dev/train \                                                                                                                                                                                                                        
  -log_path ../logs/adapter_dev/valid_giga \                                                                                                                                                                                                                       
  -result_path ../results/adapter_dev/valid_all_giga \                                                                                                                                                                                                             
  -visible_gpus 0 \                                                                                                                                                                                                                                                
  -accum_count 1 \                                                                                                                                                                                                                                                 
  -batch_size 4 \                                                                                                                                                                                                                                                  
  -deterministic_batch_size true \                                                                                                                                                                                                                                 
  -sep_optim true -use_bert_emb true -use_interval true \                                                                                                                                                                                                          
  -lr_bert 0.0002 -lr_dec 0.0002 \                                                                                                                                                                                                                                 
  -lr_bert_inner 0.0002 -lr_dec_inner 0.0002 \                                                                                                                                                                                                                     
  -train_steps 2 -report_every 1 \                                                                                                                                                                                                                                 
  -inner_train_steps 8 -report_inner_every 8 \                                                                                                                                                                                                                     
  -outer_no_warm_up true \                                                                                                                                                                                                                                         
  -inner_no_warm_up true \                                                                                                                                                                                                                                         
  -dec_adapter true \                                                                                                                                                                                                                                              
  -enc_adapter true \                                                                                                                                                                                                                                              
  -adapter_size 64 \                                                                                                                                                                                                                                               
  -meta_mode true \                                                                                                                                                                                                                                                
  -num_batch_in_task 1 \                                                                                                                                                                                                                                           
  -num_task 1 \                                                                                                                                                                                                                                                    
  -test_all \      
```

## Testing (decoding)
```
python train.py -task abs -mode test \                                                                                                                                                                                                                                                                                                                                                                                                                                                                
  -bert_data_path ../datasets/bert_datasets/shard_10/aeslc \                                                                                                                                                                                                       
  -model_path ../models/adapter_dev/train \                                                                                                                                                                                                                        
  -log_path ../logs/adapter_dev/test \                                                                                                                                                                                                                             
  -result_path ../results/adapter_dev/test \                                                                                                                                                                                                                       
  -visible_gpus 0 \                                                                                                                                                                                                                                                
  -batch_size 3000 -test_batch_size 1500 \                                                                                                                                                                                                                         
  -sep_optim true -use_interval true \                                                                                                                                                                                                                             
  -max_pos 512 -max_length 30 -min_length 5 -alpha 0.7 \                                                                                                                                                                                                           
  -enc_adapter true \                                                                                                                                                                                                                                              
  -dec_adapter true \                                                                                                                                                                                                                                              
  -adapter_size 64 \                                                                                                                                                                                                                                               
  -test_from ../models/adapter_dev/train/model_step_2.pt \             
```

# Citation
```
@inproceedings{Chen2021Meta,
  title = {Meta-Transfer Learning for Low-Resource Abstractive Summarization},
  booktitle = {Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI 2021)},
  author={Chen, Yi-Syuan and Shuai, Hong-Han},
  year = {2021},
  month = {February}
}
```

# Acknowledgements
Our implementations use the source code from the following repositories:

[Text Summarization with Pretrained Encoders](https://github.com/nlpyang/PreSumm)

