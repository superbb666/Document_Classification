###
 # @Author: Kai Niu
 # @Date: 2021-01-16 04:27:07
 # @LastEditors: Kai Niu
 # @LastEditTime: 2021-01-16 04:36:39
### 
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 256 --lstm-num-layers 1 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 64 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 512 --lstm-num-layers 1 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 64 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 256 --lstm-num-layers 2 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 64 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 512 --lstm-num-layers 2 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 64 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 256 --lstm-num-layers 1 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 128 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 512 --lstm-num-layers 1 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 128 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 256 --lstm-num-layers 2 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 128 --min-count 5 --max-vocab 10000 --embedding-dim 300
python -u train.py --epochs 40 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --lstm-hidden-size 512 --lstm-num-layers 2 --embedding_path ../data/glove/glove.6B.300d.txt --hidden-size2 128 --min-count 5 --max-vocab 10000 --embedding-dim 300