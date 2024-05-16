import pandas as pd 
import matplotlib.pyplot as plt
import torch
import argparse
import seaborn as sns
import os
import hashlib
import transformers
from transformers import pipeline
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.cluster import BisectingKMeans


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Global server clustering ")
    parser.add_argument('--data_csv', type=str, default="./FL_data", help='Number of clients for data split')
    parser.add_argument('--file_directory', type=str, default ="./archive/imagenet-10",help='upper most directory for data files')
    parser.add_argument('--file_name', type=str, default='FL_data', help='ouptput df file name')
    parser.add_argument('--max_iter', type=int, default=100, help='Bisect Kmeans iter time')
    args = parser.parse_args()

    #openning dataset
    print(f'Openning file {args.file_name}')
    data = pd.read_csv(args.data_csv)
    if data is None:
        raise ValueError(f"NO File name : {args.data_csv}")
    target_data_csv = data[data['Position']=="Server"]
    Folders = target_data_csv['Folder']
    Files = target_data_csv['File']
    img_list = []
    output_feature = []
    batch_size = 1000

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe_224= pipeline(task="image-feature-extraction", model_name="vit_base_patch16_224", device=DEVICE, pool=True)
    
    for i in range(0, len(Folders), batch_size):
        batch_folders = Folders[i:i+batch_size]
        batch_files = Files[i:i+batch_size]
        batch_imgs = []
        
        for Folder, File in zip(batch_folders, batch_files):
            path = os.path.join(Folder, File)
            path = os.path.join(args.file_directory, path)
            try:
                img = Image.open(path)
                batch_imgs.append(img)
            except Exception as e:
                print(f"Error opening image '{path}': {e}")
        
        batch_feature = pipe_224(batch_imgs)
        output_feature.append(batch_feature)
    
        # Close opened images
        for img in batch_imgs:
            img.close()

    print(f'Openning Successed!')
    if 'name' in target_data_csv.columns:
        label = target_data_csv['name']
    else:
        raise ValueError(f'No label on dataset[\'name\']')
    
    n_cluster = len(label.unique())

    # Feature Extracting
    BK = BisectingKMeans(n_clusters = n_cluster, max_iter=args.max_iter)
    BK_y = BK.fit_predict(output_feature)
    conf_matrix = confusion_matrix(label, BK_y)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted Label')
    plt.ylabel('Class')

    random_hash = hashlib.sha1(os.urandom(32)).hexdigest()[:8]  # Taking first 8 characters
    # Create the folder if it doesn't exist
    result_path = './result/'+'sever_'+random_hash + '/'
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))
    plt.close()

    total_info = "Attributes:\n"
    for attr, value in BK.__dict__.items():
        if '_X_mean' in attr:
            continue
        total_info+=f"{attr}: {value}\n"
    total_info_path = os.path.join(result_path,"./BK_Class_Info.txt")
    with open(total_info_path,"w") as file:
        file.write(total_info)
    print("All Information Saved in {result_path}!")
