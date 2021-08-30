# nohup python -u model/train.py --model nmr --epochs 30000 --saveFreq 50 --train_file ./data/nmr_C_train.csv --valid_file ./data/nmr_C_valid.csv --test_file ./data/nmr_C_test.csv --device cuda:0 --save nmr_C_June7_dft  > nmr_C_June7_dft.txt 2>&1 &


nohup python -u model/train.py --model nmr --epochs 10000 --saveFreq 20 --train_file ./data/nmr_C_train.csv --valid_file ./data/nmr_C_valid.csv --test_file ./data/nmr_C_test.csv --device cuda:0 --save nmr_C_June7_gcn  --dft_model nmr_C_June7_dft/dft_model_5000  > nmr_C_June7_gcn.txt 2>&1 &