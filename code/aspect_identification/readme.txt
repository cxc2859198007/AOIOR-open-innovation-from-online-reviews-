apt-get update
apt-get install unzip
unzip aspect_classification.zip

python -m pip install --upgrade pip
pip install transformers
pip install GPUtil
pip install scikit-learn
pip install wandb

wandb login
e952b31442e69150527143ff86329afbe94b95a8

nohup python -u aspect_classification/main.py --train > train.out 2>&1 &

nohup python -u aspect_classification/main.py --train --model_checkpoint "" > train.out 2>&1 &

nohup python -u aspect_classification/main.py --predict --checkpoint "05-31_18-48" > test.out 2>&1 &

nvidia-smi
cat train.out
cat test.out
jobs -l
kill %
clear

zip -r aspect_classification.zip aspect_classification/
unzip aspect_classification.zip

rm -rf roberta-large-finetuned-steam-reviews