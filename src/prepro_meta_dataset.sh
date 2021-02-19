#python prepro/prepro_meta_dataset.py -input_path ../datasets/bert_datasets/shard_10 -output_path ../datasets/bert_meta_datasets -abbrev rtw_40K -max_train_pt_files 4000 -max_valid_pt_files 4000 -train_dataset_list reddit,reddit_tifu,wikihow -valid_dataset_list aeslc

#python prepro/rename_train_to_valid.py -input_path ../datasets/bert_meta_datasets/meta_data_rtw_40K/valid
