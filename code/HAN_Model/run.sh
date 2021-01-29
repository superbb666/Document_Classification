###
 # @Author: Kai Niu
 # @Date: 2021-01-16 03:59:07
 # @LastEditors: Kai Niu
 # @LastEditTime: 2021-01-16 04:50:06
### 

python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=256 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=512 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=256 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=512 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=256 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.2 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=512 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.2 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=256 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.2 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=512 --sent_gru_num_layers=1 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.2 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=256 --sent_gru_num_layers=2 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=512 --sent_gru_num_layers=2 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=256 --sent_gru_num_layers=2 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=512 --sent_gru_num_layers=2 --word_gru_num_layers=1 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=256 --sent_gru_num_layers=1 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=512 --sent_gru_num_layers=1 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=256 --sent_gru_num_layers=1 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=512 --sent_gru_num_layers=1 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=256 --sent_gru_num_layers=2 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=256 --sent_gru_hidden_dim=512 --sent_gru_num_layers=2 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=256 --sent_gru_num_layers=2 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 
python train.py --vocab_path=../data/glove/glove.6B.300d.txt --num_epochs=30 --lr=1e-3 --embed_dim=300 --word_gru_hidden_dim=512 --sent_gru_hidden_dim=512 --sent_gru_num_layers=2 --word_gru_num_layers=2 --use_layer_norm=True --dropout=0.5 