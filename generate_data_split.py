import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Data file split for Federated Learning ")
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients for data split')
    parser.add_argument('--server_size', type=int, default=700, help='server dataset size for each class')
    parser.add_argument('--file_directory', type=str, default ="./archive/imagenet-10",help='upper most directory for data files')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for dataset split')
    parser.add_argument('--file_name', type=str, default='FL_data', help='ouptput df file name')
    parser.add_argument('--data_name',type= str, default='Imagenet-10', help='name of dataset')
    args = parser.parse_args()

    data = []
    directory = args.file_directory
    np.random.seed(args.random_seed)

    client_name= []
    for i in range(args.num_clients):
        client_name.append(f'Client{i}')

    for class_name in os.listdir(directory):
        folder_path = os.path.join(directory,class_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if os.path.isfile(file_path):
                    data.append({'Folder': class_name, 'File': file_name})
    data_file = pd.DataFrame(data)

    data_file['Position'] = None
    folders = data_file['Folder'].unique()


    # Datset Label
    label = data_file['Folder']

    if args.data_name == "Imagenet-10":
        classes = label.unique()
        names = ['Penguin','Dog','Cheetah','Plane', 'Zeppelin','Ship','SoccerBall','Car','Truck','Orange']
        name_dict = {cls: name for cls, name in zip(classes, names)}
        data_file['name'] = [name_dict[name] for name in data_file['Folder']]


    for folder in folders:
        target_file = data_file[data_file['Folder']==folder]
        clients, server = train_test_split(target_file, test_size = args.server_size, random_state= args.random_seed)
        
        data_file.loc[server.index, 'Position'] = 'Server'

        clients_index=clients.index.to_list()
        np.random.shuffle(clients_index)
        split_point = np.array_split(clients_index, args.num_clients)

        for set_name, point in zip(client_name, split_point):
            data_file.loc[point,'Position'] = set_name

    data_file.to_csv(f'./{args.file_name}',index=False)
    print(f'File {args.file_name} successfully created!')
        