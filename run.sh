cd datasets/
# Thay đổi đường dẫn trong 2 file thành đường dẫn tới dataset
python crop_Reid_training_AIC23.py
python gen_Reid_training_xml_AIC23.py

# cd ../
# python train.py --config_file configs/stage1/resnext101_ibn_a.yml