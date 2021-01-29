###
 # @Author: Kai Niu
 # @Date: 2021-01-15 23:34:51
 # @LastEditors: Kai Niu
 # @LastEditTime: 2021-01-16 04:43:26
### 

python -u train.py --epochs 30 --batch-size 64 --lr 0.001 --dropout-rate 0.5 --lstm-hidden-size 256 --lstm-num-layers 1 --embedding_path ../data/glove/glove.6B.300d.txt
python -u train.py --epochs 30 --batch-size 64 --lr 0.001 --dropout-rate 0.5 --lstm-hidden-size 512 --lstm-num-layers 1 --embedding_path ../data/glove/glove.6B.300d.txt
python -u train.py --epochs 30 --batch-size 64 --lr 0.001 --dropout-rate 0.5 --lstm-hidden-size 256 --lstm-num-layers 2 --embedding_path ../data/glove/glove.6B.300d.txt
python -u train.py --epochs 30 --batch-size 64 --lr 0.001 --dropout-rate 0.5 --lstm-hidden-size 512 --lstm-num-layers 2 --embedding_path ../data/glove/glove.6B.300d.txt