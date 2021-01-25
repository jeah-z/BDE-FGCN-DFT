
mkdir nmr_H_results
python -u model/train.py --model nmr \
--epochs 2000 \
--saveFreq 10 \
--train_file ./data/nmr_H_train.csv \
--valid_file ./data/nmr_H_valid.csv \
--test_file ./data/nmr_H_test.csv \
--device cuda:0 --save nmr_H_results
