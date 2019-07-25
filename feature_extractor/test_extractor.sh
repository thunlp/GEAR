CUDA_VISIBLE_DEVICES=0 python extractor.py \
    --input_file ../data/gear/gear-test-set-0_001.tsv \
    --output_file ../data/gear/gear-test-set-0_001-features.tsv \
    --bert_model ../pretrained_models/BERT-Pair/ \
    --do_lower_case \
    --max_seq_length 128 \
    --batch_size 512 \