#############
##  AESLC  ##
#############
#python prepro/prepro_tfds_dataset.py -mode download -dataset aeslc -output_path ../datasets/organized_raw_datasets/ -art_feature_name email_body -summ_feature_name subject_line
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/aeslc/ -save_path ../datasets/tokenized_dataset/aeslc/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/aeslc/ -save_path ../datasets/analysis_dataset/aeslc/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/aeslc/ -save_path ../datasets/json_dataset/aeslc/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/aeslc/ -save_path ../datasets/bert_datasets/shard_10/aeslc/ -max_src_ntokens 512 -max_tgt_ntokens 32
#
################
##  BIGPATENT ##
################
#python prepro/prepro_tfds_dataset.py -mode download -dataset big_patent -output_path ../datasets/organized_raw_datasets/ -art_feature_name text -summ_feature_name summary
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/big_patent/ -save_path ../datasets/tokenized_dataset/big_patent/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/big_patent/ -save_path ../datasets/analysis_dataset/big_patent/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/big_patent/ -save_path ../datasets/json_dataset/big_patent/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/big_patent/ -save_path ../datasets/bert_datasets/shard_10/big_patent/  -max_src_ntokens 512 -max_tgt_ntokens 256
#
###############
##  Billsum  ##
###############
#python prepro/prepro_tfds_dataset.py -mode download -dataset billsum -output_path ../datasets/organized_raw_datasets/ -art_feature_name text -summ_feature_name summary
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/billsum/ -save_path ../datasets/tokenized_dataset/billsum/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/billsum/ -save_path ../datasets/analysis_dataset/billsum/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/billsum/ -save_path ../datasets/json_dataset/billsum/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/billsum/ -save_path ../datasets/bert_datasets/shard_10/billsum/  -max_src_ntokens 512 -max_tgt_ntokens 256
#
####################
## Cnn Dailymail  ##
####################
#python prepro/prepro_tfds_dataset.py -mode download -dataset cnn_dailymail -output_path ../datasets/organized_raw_datasets/ -art_feature_name article -summ_feature_name highlights
#python prepro/prepro_tfds_dataset.py -mode ssplit -dataset cnn_dailymail -output_path ../datasets/organized_raw_datasets/ -ssplit_target art
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/cnn_dailymail/ -save_path ../datasets/tokenized_dataset/cnn_dailymail/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/cnn_dailymail/ -save_path ../datasets/analysis_dataset/cnn_dailymail/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/cnn_dailymail/ -save_path ../datasets/json_dataset/cnn_dailymail/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/cnn_dailymail/ -save_path ../datasets/bert_datasets/shard_10/cnn_dailymail/  -max_src_ntokens 512 -max_tgt_ntokens 256
#
###############
##  Gigaword ##
###############
#python prepro/prepro_tfds_dataset.py -mode download -dataset gigaword -output_path ../datasets/organized_raw_datasets/ -art_feature_name document -summ_feature_name senummary
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/gigaword/ -save_path ../datasets/tokenized_dataset/gigaword/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/gigaword/ -save_path ../datasets/analysis_dataset/gigaword/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/gigaword/ -save_path ../datasets/json_dataset/gigaword/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/gigaword/ -save_path ../datasets/bert_datasets/shard_10/gigaword/  -max_src_ntokens 128 -max_tgt_ntokens 32
#
#################
##  MultiNews  ##
#################
#python prepro/prepro_tfds_dataset.py -mode download -dataset multi_news -output_path ../datasets/organized_raw_datasets/ -art_feature_name document -summ_feature_name summary
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/multi_news/ -save_path ../datasets/tokenized_dataset/multi_news/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/multi_news/ -save_path ../datasets/analysis_dataset/multi_news/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/multi_news/ -save_path ../datasets/json_dataset/multi_news/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/multi_news/ -save_path ../datasets/bert_datasets/shard_10/multi_news/  -max_src_ntokens 512 -max_tgt_ntokens 256
#
################
##  NewsRoom  ##
################
#python prepro/prepro_tfds_dataset.py -mode download -dataset newsroom -output_path ../datasets/organized_raw_datasets/ -art_feature_name text -summ_feature_name summary
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/newsroom/ -save_path ../datasets/tokenized_dataset/newsroom/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/newsroom/ -save_path ../datasets/analysis_dataset/newsroom/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/newsroom/ -save_path ../datasets/json_dataset/newsroom/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/newsroom/ -save_path ../datasets/bert_datasets/shard_10/newsroom/ -max_src_ntokens 512 -max_tgt_ntokens 128
#
##############
##  Reddit  ##
##############
#python prepro/prepro_tfds_dataset.py -mode download -dataset reddit -output_path ../datasets/organized_raw_datasets/ -art_feature_name content -summ_feature_name summary
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/reddit/ -save_path ../datasets/tokenized_dataset/reddit/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/reddit/ -save_path ../datasets/analysis_dataset/reddit/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/reddit/ -save_path ../datasets/json_dataset/reddit/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/reddit/ -save_path ../datasets/bert_datasets/shard_10/reddit/ -max_src_ntokens 512 -max_tgt_ntokens 128

###################
##  Reddit-TIFU  ##
###################
#python prepro/prepro_tfds_dataset.py -mode download -dataset reddit_tifu -output_path ../datasets/organized_raw_datasets/ -art_feature_name documents -summ_feature_name title
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/reddit_tifu/ -save_path ../datasets/tokenized_dataset/reddit_tifu/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/reddit_tifu/ -save_path ../datasets/analysis_dataset/reddit_tifu/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/reddit_tifu/ -save_path ../datasets/json_dataset/reddit_tifu/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/reddit_tifu/ -save_path ../datasets/bert_datasets/shard_10/reddit_tifu/ -max_src_ntokens 512 -max_tgt_ntokens 128

#########################
##  Scientific Papers  ##
#########################
#python prepro/prepro_tfds_dataset.py -mode download -dataset scientific_papers/pubmed -output_path ../datasets/organized_raw_datasets -art_feature_name article -summ_feature_name abstract
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/scientific_papers/pubmed/ -save_path ../datasets/tokenized_dataset/scientific_papers/pubmed/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasetsa/tokenized_dataset/scientific_papers/pubmed/ -save_path ../datasets/analysis_dataset/scientific_papers/pubmed/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/scientific_papers/pubmed/ -save_path ../datasets/json_dataset/scientific_papers_pubmed/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/scientific_papers_pubmed/ -save_path ../datasets/bert_datasets/shard_10/scientific_papers_pubmed/ -max_src_ntokens 512 -max_tgt_ntokens 256
#
#python prepro/prepro_tfds_dataset.py -mode download -dataset scientific_papers/arxiv -output_path ../datasets/raw_datasets -art_feature_name article -summ_feature_name abstract 
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/scientific_papers/arxiv/ -save_path ../datasets/tokenized_dataset/scientific_papers/arxiv/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/scientific_papers/arxiv/ -save_path ../datasets/analysis_dataset/scientific_papers/arxiv/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/scientific_papers/arxiv/ -save_path ../datasets/json_dataset/scientific_papers_arxiv/ -shard_size 10 
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/scientific_papers_arxiv/ -save_path ../datasets/bert_datasets/shard_10/scientific_papers_arxiv/ -max_src_ntokens 512 -max_tgt_ntokens 256
#
###############
##  WikiHow  ##
###############
#python prepro/prepro_tfds_dataset.py -mode download -dataset wikihow -output_path ../datasets/organized_raw_datasets/ -art_feature_name text -summ_feature_name headline
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/wikihow/ -save_path ../datasets/tokenized_dataset/wikihow/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/wikihow/ -save_path ../datasets/analysis_dataset/wikihow/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/wikihow/ -save_path ../datasets/json_dataset/wikihow/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/wikihow/ -save_path ../datasets/bert_datasets/shard_10/wikihow/ -max_src_ntokens 512 -max_tgt_ntokens 256
#
############
##  XSum  ##
############
#python prepro/prepro_tfds_dataset.py -mode download -dataset xsum -output_path ../datasets/organized_raw_datasets/ -art_feature_name article -summ_feature_name highlights
#python preprocess.py -mode tokenize_tfds -raw_path ../datasets/organized_raw_datasets/xsum/ -save_path ../datasets/tokenized_dataset/xsum/ -limit_num_file_tokenized 40000
#python preprocess.py -mode dataset_analysis_tfds -raw_path ../datasets/tokenized_dataset/xsum/ -save_path ../datasets/analysis_dataset/xsum/
#python preprocess.py -mode format_to_lines_tfds -raw_path ../datasets/tokenized_dataset/xsum/ -save_path ../datasets/json_dataset/xsum/ -shard_size 10
#python preprocess.py -mode format_to_bert -raw_path ../datasets/json_dataset/xsum/ -save_path ../datasets/bert_datasets/shard_10/xsum/  -max_src_ntokens 512 -max_tgt_ntokens 256
