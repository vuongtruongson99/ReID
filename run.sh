cd datasets/
# Thay đổi đường dẫn trong 2 file thành đường dẫn tới dataset
python crop_Reid_training_AIC23.py
python gen_Reid_training_xml_AIC23.py

cd ../
# ResNext101-IBN-a: https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth (pretrained imagenet)
wget https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth
mv resnext101_ibn_a-6ace051d.pth pretrained
python train.py --config_file configs/stage1/resnext101_ibn_a.yml
python test.py --config_file configs/stage2/resnext101_ibn_a.yml TEST.WEIGHT 'logs/stage2/resnext101a_384_AIC23/v1/resnext101_ibn_a_2.pth' OUTPUT_DIR 'logs/stage2/resnext101a_384_AIC23/v1'

# # ResNet101-IBN-a: https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth (pretrained imagenet)
# wget https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth
# mv resnet101_ibn_a-59ea0ac6.pth pretrained
# python train.py --config_file configs/stage1/resnet101_ibn_a.ymlresnext101a_384