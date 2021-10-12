import cv2
import os
data_dir='/home/audience/dataset/mscoco/train2/train_class'
data2_dir='/home/audience/dataset/mscoco/val2/val_class'
output_dir='/home/audience/dataset/mscoco/train3/train_class'
output2_dir='/home/audience/dataset/mscoco/val4/val_class'
i=0
for file in os.listdir(data2_dir):
    i=i+1
    if i<=200:
        print(i)
        file_dir=data2_dir+'/'+file
        img=cv2.imread(file_dir)
        output2_file=output2_dir+'/'+file
        cv2.imwrite(output2_file,img)

