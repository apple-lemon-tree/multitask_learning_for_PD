# A Personality Detection Model Based on Data Augmentation and Multi-Task Learning

## Step 1: Label Decoupling and GPT-based Data Generation
**Objective:** Decouple the labels and utilize GPT to generate augmented personality data.

Before generating data, you must first train a **filter model** to screen the generated content in subsequent steps.

### 1.1 Train the filter model
```bash
python DataAugment/train.filter_model.py \
  --dataset "Essays" \
  --train_epochs 30 \
  --seed 42 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --max_len 512 \
  --model_url "bert-base-uncased"
```

### 1.2 Generate new data
Once the filter is trained, execute data_generation.py to generate the synthetic dataset.
```bash
python DataAugment/data_generation.py \
  --dataset "Essays" \
  --num_train_epochs 30 \
  --seed 42 \
  --batch_size 4 \
  --alpha 0.5 \
  --beta 0.5 \
  --bert_lr 1e-5 \
  --bert_max_len 512 \
  --gpt_max_len 512 \
  --gpt_top_p 0.95 \
  --gpt_tok_k 50
```

## Step 2: Back-Translation for Rare Classes
**Objective:** Perform back-translation on scarce/rare data samples to balance the dataset.
 
### 2.1 Prerequisites
Before running the back-translation script, you must download the affectivespace.csv file from the official SenticNet website:  https://sentic.net/downloads/

### 2.2 Exexute Back-Translation
Run the following script to process low-frequency data:
```bash
python DataAugment/back_transaltion/back_trans.py
```

### 2.3 Filter by Similarity
Use data_similarity.py to filter out samples where semantic features and psycholinguistic features do not meet the specified thresholds.

## Step 3: Multi-Task Learning Model Training
**Objective:** Train the final multi-task learning model using the augmented dataset.
```bash
python multitask/multitask_1.py
```
