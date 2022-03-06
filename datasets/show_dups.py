import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

df = pd.read_csv('duplicates.csv', index_col=0)
output_path = '/home/mborisov/CLM/duplicates'
os.makedirs(output_path, exist_ok=True)

for i, row in df.iterrows():

    img1, img2 = row['img1'], row['img2']

    print(img1)
    print(img2)

    img1 = cv2.imread(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(img1)
    axs[1].imshow(img2)

    plt.savefig(os.path.join(output_path, f'{i}.jpg'))
