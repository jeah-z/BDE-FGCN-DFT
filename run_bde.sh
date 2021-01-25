mkdir bde_CH_results
python -u model/train.py \
--model bde \
--epochs 2000 \
--saveFreq 10 \
--train_file ./data/bde_CH_train.csv \
--valid_file ./data/bde_CH_valid.csv \
--test_file ./data/bde_CH_test.csv \
--device cuda:0 \
--save bde_CH_results