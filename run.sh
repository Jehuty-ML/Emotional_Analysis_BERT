==FC结构
# 仅训练
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_train=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name fc

# 仅评估
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_eval=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name fc

# 仅预测
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_predict=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name fc

# 先训练、再评估、最后预测
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_train=True --do_eval=True --do_predict=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name=fc

==BiLSTM结构
# 仅训练
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_train=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name=bilstm

# 仅评估
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_eval=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name bilstm

# 仅预测
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_predict=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name bilstm

# 先训练、再评估
python run_emotional_analysis.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_train=True --do_eval=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0 --cls_model_name bilstm