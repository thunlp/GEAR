## GEAR

Source code and dataset for the ACL 2019 paper "[GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification](GEAR.pdf)".

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

Our evidence extraction results can be found in [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1499a062447f4a3d8de7/) or [Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K).

Download these files and put them in the ``data/retrieved/`` folder. Then the folder will look like

```
data/retrieved/
    train.ensembles.s10.jsonl
    dev.ensembles.s10.jsonl
    test.ensembles.s10.jsonl
```

## Data Preparation
```
# Download the fever database
wget -O data/fever/fever.db https://s3-eu-west-1.amazonaws.com/fever.public/wiki_index/fever.db

# Extract the evidence from database
cd scripts/
python retrieval_to_bert_input.py

# Build the datasets for gear
python build_gear_input_set.py

cd ..
```

## Feature Extraction
First download our pretrained BERT-Pair model ([Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/1499a062447f4a3d8de7/?p=/BERT-Pair&mode=list) or [Google Cloud](https://drive.google.com/drive/folders/1y-5VdcrqEEMtU8zIGcREacN1JCHqSp5K)) and put the files into the ``pretrained_models/BERT-Pair/`` folder.

Then the folder will look like this:
```
pretrained_models/BERT-Pair/
    	pytorch_model.bin
    	vocab.txt
    	bert_config.json
```

Then run the feature extraction scripts.
```
cd feature_extractor/
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
cd ..
```

## GEAR Testing
```
cd gear
CUDA_VISIBLE_DEVICES=0 python test.py
cd ..
```

## Results Gathering
```
cd gear
python results_scorer.py
cd ..
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