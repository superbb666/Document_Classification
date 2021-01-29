###
 # @Author: Kai Niu
 # @Date: 2021-01-16 04:13:10
 # @LastEditors: Kai Niu
 # @LastEditTime: 2021-01-16 04:22:34
### 
python -u train.py --epochs 20 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --max-vocab 10000 --min-count 5 --embedding_path ../data/glove/glove.6B.300d.txt
python -u train.py --epochs 20 --batch-size 64 --lr 0.01 --dropout-rate 0.4 --max-vocab 10000 --min-count 5 --embedding_path ../data/glove/glove.6B.300d.txt
python -u train.py --epochs 20 --batch-size 64 --lr 0.01 --dropout-rate 0.3 --max-vocab 10000 --min-count 5 --embedding_path ../data/glove/glove.6B.300d.txt