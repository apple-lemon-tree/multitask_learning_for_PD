# A Personality Detection Model Based on Data Augmentation and Multi-Task Learning
First
对标签进行解耦，使用GPT生成新的人格数据对标签进行解耦，使用GPT生成新的人格数据
你需要先训练一个过滤器以便后续筛选生成的数据
python DataAugment/train.filter_model.py --dataset "Essays" --train_epochs 30 --seed 42 --batch_size 4 --learning_rate 1e-5 --max_len 512 --model_url "bert-base-uncased"
然后，你就可以执行data_generation.py文件来生成数据
python DataAugment/data_generation.py --dataset "Essays" --num_train_epochs 30 --seed 42 --batch_size 4 --alpha 0.5 --beta 0.5 --bert_lr 1e-5 --bert_max_len 512 --gpt_max_len 512 --gpt_top_p 0.95 --gpt_tok_k 50

Second
对更稀有的数据进行回译操作，在这之前，你需要去Senticnet官网上下载affectivespace.csv文件   https://sentic.net/downloads/
然后你就可以使用back_trans.py来对低频数据进行处理
python DataAugment/back_transaltion/back_trans.py
你可以用
