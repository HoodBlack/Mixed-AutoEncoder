import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

def calculate_entropy(series):
    counts = series.value_counts()
    probabilities = counts / counts.sum()
    return entropy(probabilities)

def calculate_imbalance_ratio(series):
    counts = series.value_counts()
    return counts.max() / counts.min()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Data file split for Federated Learning ")
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients for data split')
    parser.add_argument('--server_size', type=int, default=700, help='server dataset size for each class')
    parser.add_argument('--file_directory', type=str, default ="./archive/imagenet-10",help='upper most directory for data files')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for dataset split')
    parser.add_argument('--file_name', type=str, default='FL_data', help='ouptput df file name')
    parser.add_argument('--data_name',type= str, default='Imagenet-10', help='name of dataset')
    parser.add_argument('--mode', type = str, default = "uniform", help='data distribution')
    args = parser.parse_args()

    data = []
    directory = args.file_directory
    # random seed
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
    total_n = len(data_file)
    data_file['Position'] = None
    folders = data_file['Folder'].unique()


    # Datset Label
    label = data_file['Folder']

    if args.data_name == "Imagenet-10":
        classes = label.unique()
        names = ['Penguin','Dog','Cheetah','Plane', 'Zeppelin','Ship','SoccerBall','Car','Truck','Orange']
        name_dict = {cls: name for cls, name in zip(classes, names)}
        data_file['name'] = [name_dict[name] for name in data_file['Folder']]

    if args.mode == 'unifrom':
        print("mode - uniform")
        for folder in folders:
            target_file = data_file[data_file['Folder']==folder]
            clients, server = train_test_split(target_file, test_size = args.server_size, random_state= args.random_seed)
            
            data_file.loc[server.index, 'Position'] = 'Server'

            clients_index=clients.index.to_list()
            np.random.shuffle(clients_index)
            split_point = np.array_split(clients_index, args.num_clients)

            for set_name, point in zip(client_name, split_point):
                data_file.loc[point,'Position'] = set_name
    
    if args.mode == 'non-uniform':
        print("mode - non-uniform")
        client_data= []
        for folder in folders:
            target_file = data_file[data_file['Folder']==folder]
            clients, server = train_test_split(target_file, test_size = args.server_size, random_state= args.random_seed)
            data_file.loc[server.index, 'Position'] = 'Server'

            client_data.extend(clients.index.to_list())

        client_data_n = len(client_data)
        np.random.shuffle(client_data)
        split_point = np.array_split(client_data, args.num_clients)

        for set_name, point in zip(client_name,split_point):
            data_file.loc[point,'Position'] = set_name

        for c in range(args.num_clients):
            target_file = data_file[data_file['Position'] == client_name[c]]
            print(target_file['name'].value_counts())
            # ent = calculate_entropy(target_file['name'])
            # print(f"Client {client_name[c]} 'name' entropy: {ent}\n")
            imbalance_ratio = calculate_imbalance_ratio(target_file['name'])
            print(f"Client {client_name[c]} 'name' imbalance ratio: {imbalance_ratio}\n")
                

        

    data_file.to_csv(f'./{args.file_name}',index=False)
    print(f'File {args.file_name} successfully created!')
        