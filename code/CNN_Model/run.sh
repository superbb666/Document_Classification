###
 # @Author: Kai Niu
 # @Date: 2021-01-16 04:13:10
 # @LastEditors: Kai Niu
 # @LastEditTime: 2021-01-16 04:15:10
### 
python -u train.py --epochs 25 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --filter-size 64 --kernel-size '2,3,4'
python -u train.py --epochs 25 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --filter-size 128 --kernel-size '2,3,4'
python -u train.py --epochs 25 --batch-size 64 --lr 0.01 --dropout-rate 0.5 --filter-size 256 --kernel-size '2,3,4'
python -u train.py --epochs 25 --batch-size 64 --lr 0.01 --dropout-rate 0.4 --filter-size 128 --kernel-size '2,3,4'