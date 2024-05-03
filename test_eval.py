import numpy as np

def monti_like_evaluation(data, label):
    """
    Parameters
    ----------
    data
        PCA result vectors, 1-D shape
    label
        answer array
    """
    N = len(label)
    adj_m, ans_m = create_matrix(data,label)
    eval_matrix = ans_m /(adj_m)
    eval_matrix = np.nan_to_num(eval_matrix, nan=0, posinf=0, neginf=0)
    result_num = np.count_nonzero(eval_matrix==1)
    print(f'evalutaion value : {(result_num-N)/(2*N)}')
    

def create_matrix(data, label):
    # the input is raw data. 
    if len(data.shape)!=1:
        raise ValueError("Input data must be a 1-dimensional array.")
    N = len(data)
    adj_matrix = np.zeros((N,N))
    ans_matrix = np.zeros((N,N))
    for row in range(N):
        for col in range(N):
            if data[row] == data[col]:
                adj_matrix[row,col] = 1
            if label[row]==label[col]:
                ans_matrix[row,col]=1
    return adj_matrix, ans_matrix