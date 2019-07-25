## GEAR

Source code and dataset for the ACL 2019 paper "GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification".

## Requirements:
Please make sure your environment includes:
```
python (tested on 3.6.7)
pytorch (tested on 1.0.0)
```
Then, run the command:
```
pip install -r requirements.txt
```

## Evidence Extraction
We use the codes from [Athene UKP TU Darmstadt](https://github.com/UKPLab/fever-2018-team-athene) in the document retrieval and sentence selection steps. 

Our evidence extraction results can be found in the /data/retrieved/ folder.

## Data Preparation
```
# Download the fever database
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db

# Extract the evidence from database
cd scripts/
python retrieval_to_bert_input.py

# Build the datasets for gear
python build_gear_input_set.py
```

## Feature Extraction
```
cd ../feature_extractor/
chmod +x *.sh
./train_extracor.sh
./dev_extractor.sh
./test_extractor.sh
cd ..
```

## GEAR Training
```
cd gear
CUDA_VISIBLE_DEVICES=0 python train.py
```

## GEAR Testing
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

## Results Gathering
```
python results_scorer.py
```

## Cite

If you use the code, please cite our paper:

```
@inproceedings{zhou2019gear,
  title={GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification},
  author={Zhou, Jie and Han, Xu and Yang, Cheng and Liu, Zhiyuan and Wang, Lifeng and Li, Changcheng and Sun, Maosong},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```