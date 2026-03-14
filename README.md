# MSWCF:A multisource weakly supervised high-resolution crop mapping framework based on low-resolution noise labels

# Specifications of Input Data for Model Training

1. An image file where the first 4 bands correspond to GF super-resolution 2 m imagery (in the order: Blue, Green, Red, NIR), and the subsequent 6 bands correspond to upsampled 2 m Sentinel 2 imagery (in the order: RE1, RE2, RE3, B8A, SWIR1, SWIR2).

2. The data type is unsigned 16 bit integer.

3. The model consists of two branches, reading the first 4 bands and the last 6 bands respectively.

4. Generate training and test lists (.csv) for the dataset (sample lists are provided in the "dataset" folder named "crop.csv")


# Training Instructions

1. Download the imagenet21k ViT pre-train model at [**Pre-train ViT**](https://drive.google.com/file/d/10Ao75MEBlZYADkrXE4YLg6VObvR0b2Dr/view?usp=sharing) and put it at *"./networks/pre-train_model/imagenet21k"*
   
2. Put the training data at *"./dataset/cropdataset"*.
   
3. Run the "Train" command:
   
   python train_multi_model.py.py --dataset cropdataset --batch_size 36 --max_epochs 100 --savepath *save path of your folder* --gpu 0,1
4. After training, run the "Test" command:
   
   python test_multi_model.py --dataset cropdataset --model_path *The path of trained .pth file* --save_path *To save the inferred results* --gpu 0,1
   








