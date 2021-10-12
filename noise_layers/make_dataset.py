import cv2
data_dir='/home/audience/dataset/mscoco/train2/train_class'
output_dir='/home/audience/dataset/mscoco/train3/train_class'
i=0
for file in data_dir:
    i=i+1
    if i<=5000:
        file_dir=data_dir+file
        img=cv2.imread(file_dir)
        output_file=output_dir+file
        cv2.imwrite(output_file,img)

