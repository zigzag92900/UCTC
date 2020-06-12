usage: python run.py [-h] [--dataset DATASET] [--batch BATCH] [--epoch EPOCH] [--lr LR] [--up UP] [--dw DW] [--period PERIOD] [--type TYPE]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  UCR univariate dataset (default: CBF)
  --batch BATCH      batch size (default: 128)
  --epoch EPOCH      number of iteration (default: 200)
  --lr LR            learning rate (default: 0.001)
  --up UP            upper bound of threshold for DLG (default: 1.0)
  --dw DW            lower bound of threshold for DLG (default: 0.8)
  --period PERIOD    period of updating threshold for DLG (default: 10)
  --type TYPE        type of encoder; C for CNN, R for RNN, CR for CNN+RNN (default: CR)

example:
python run.py --dataset TwoPatterns --batch 128 --epoch 200 --lr 0.001 --up 1 --dw 0.75
python run.py --dataset Symbols --batch 166 --epoch 200 --lr 0.001 --up 1 --dw 0.75
python run.py --dataset Plane --batch 128 --epoch 200 --lr 0.0005 --up 1 --dw 0.80

Python 3.6.10 and Torch 1.5.0 is used.
More data is available on http://www.timeseriesclassification.com
http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip