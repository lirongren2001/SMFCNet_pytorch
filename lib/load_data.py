import os
import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle

def sym_wsr(data,n_subjects=125, n_regions=120, n_lambda=11):
    '''
        Symmetrization of WSR.
    '''
    # 创建一个用于存储对称化后的数据的数组
    _sym_data = np.zeros((n_lambda,n_subjects,n_regions,n_regions))
    # 对每个参数lambda进行循环
    for param_lambda1 in range(n_lambda):
        # 获取对应lambda值下的特征数据
        tmpFeature = data[param_lambda1]
        # 创建一个用于存储对称化后的网络的数组
        sym_Net = np.zeros((n_subjects,n_regions,n_regions))
        # 对每个受试者进行循环
        for i in range(n_subjects):
            # 获取原始网络
            originalNet = tmpFeature[i]
            # 将原始网络与其转置的平均值作为对称化后的网络
            originalNet = (originalNet + originalNet.T) / 2
            # 将对称化后的网络存储到数组中
            sym_Net[i,:] = originalNet
        # 将对称化后的网络存储到对称化数据数组中
        _sym_data[param_lambda1, :] = sym_Net
    return _sym_data

# 定义了两个全局变量用于后续函数中的默认值
nor_num =
pat_num =

def load_data(dataFile,n_subjects, n_regions, n_lambda, nor_num = nor_num, pat_num = pat_num):
    '''
    :param dataFile: Sparsity brain network数据的位置。
    :param n_subjects: 所有受试者数量。
    :param n_regions: 所有脑区域数量。
    :param n_lambda: 稀疏度参数。
    :return: 所有的数据和标签。
    '''
    # 加载.mat文件中的数据
    data = sio.loadmat(dataFile)
    # 获取.mat文件中名为"BrainNetSet"的数据
    BrainNetSet = data["BrainNetSet"]
    # 创建一个空元组用于存储不同lambda值下的脑网络数据
    tupleBrainNetSet = ()
    # 对每个lambda值进行循环
    for i in range(n_lambda):
        # 将.brain网络数据转换为numpy数组，并且reshape为适合存储的形状
        a = np.array(list(BrainNetSet[i])).reshape(n_subjects, n_regions, n_regions)
        # 将处理后的数据添加到元组中
        tupleBrainNetSet += (a,)
    # 创建用于存储标签的数组，包括nor_num个0和pat_num个1
    nor = np.zeros((nor_num,),dtype = int)
    pat = np.ones((pat_num,),dtype = int)
    label = np.concatenate((nor,pat))
    # 将元组转换为numpy数组
    arrayBrainNetSet = np.array((tupleBrainNetSet))

    # 如果数据文件是SZ_WSR.mat，则对数据进行绝对值处理后进行对称化处理
    if dataFile == r'data\BrainNetSet_SZ_WSR.mat':
        arrayBrainNetSet = np.abs(sym_wsr(arrayBrainNetSet))  

    return arrayBrainNetSet,label

def threshold(data,n_subjects, n_regions, n_lambda):
    '''
    wsr的二值化。
    '''
    # 创建一个用于存储最终阈值后的数据的数组
    threshold_final = np.zeros((n_lambda, n_subjects, n_regions, n_regions))
    # 对每个lambda值进行循环
    for i in range(n_lambda):
        # 创建一个用于存储当前lambda值下的阈值后的数据的数组
        threshold = np.zeros((n_subjects, n_regions, n_regions))
        # 对每个受试者的脑网络数据进行循环
        for j in range(n_subjects):
            # 获取当前lambda值和受试者的脑网络数据
            sub_wsr = data[i][j]
            # 将非零元素转换为1，零元素转换为0，实现二值化
            _single_threshold = (sub_wsr != 0).astype(int)
            # 将阈值后的数据存储到数组中
            threshold[j,:] = _single_threshold
        # 将当前lambda值下的阈值后的数据存储到最终数组中
        threshold_final[i,:] = threshold
    return threshold_final # n_lambda, n_subjects, n_regions, n_regions

def sparse_guided(data_wsr_threshold,data_single_PC, n_subjects, n_regions, n_lambda):
    '''
    PC由二值化的WSR稀疏引导。
    '''
    # 创建一个用于存储对PC稀疏引导后的数据的数组
    threshold_for_PC = np.zeros((n_lambda, n_subjects, n_regions, n_regions))
    # 对每个lambda值进行循环
    for i in range(n_lambda):
        # 创建一个用于存储当前lambda值下的PC稀疏引导后的数据的数组
        _PC_ = np.zeros((n_subjects, n_regions, n_regions))
        # 对每个受试者进行循环
        for j in range(n_subjects):
            # 获取当前lambda值下的WSR阈值后的数据和单个受试者的PC数据
            s_threshold = data_wsr_threshold[i][j]
            s_pc = data_single_PC[j]
            # 将WSR阈值数据和PC数据相乘，实现稀疏引导
            _wsr_thred_PC = s_threshold * s_pc
            # 将稀疏引导后的PC数据存储到数组中
            _PC_[j, :] = _wsr_thred_PC
        # 将当前lambda值下的PC稀疏引导后的数据存储到最终数组中
        threshold_for_PC[i, :] = _PC_
    return threshold_for_PC

def transform(sparse_guided_matrix,n_subjects, n_regions, n_lambda):
    '''
    :return: 转换后的矩阵
    '''
    # 获取除了第一个lambda值外的稀疏引导矩阵
    threshold_ten = sparse_guided_matrix[1:, :]
    # 创建一个用于存储转换后的数据的数组
    threshold_change = np.zeros((n_subjects,n_lambda-1, n_regions, n_regions))
    # 对每个受试者进行循环
    for ii in range(n_subjects):
        # 获取每个受试者的稀疏引导矩阵
        b = threshold_ten[:, ii, :]
        # 将稀疏引导矩阵存储到转换后的数据数组中
        threshold_change[ii, :] = b
    return threshold_change

def load_SMFCN_label(n_subjects = 125,n_regions = 120, n_lambda = 11):
    # 加载两个不同文件中的数据和标签
    dataFile_PC = r'data\BrainNetSet_HC_SZ_PC.mat'
    dataFile_WSR = r'data\BrainNetSet_SZ_WSR.mat'
    # 分别加载两种类型的数据和标签
    pc, label = load_data(dataFile_PC,n_subjects,n_regions,n_lambda)
    wsr, label = load_data(dataFile_WSR,n_subjects,n_regions,n_lambda)
    # 获取第一个lambda值下的PC数据
    _single_pc = pc[0]
    # 对WSR数据进行阈值化处理
    binarization_wsr = threshold(wsr,n_subjects,n_regions,n_lambda)
    # 对阈值化后的WSR数据和第一个lambda值下的PC数据进行稀疏引导处理
    _sparse_guided = sparse_guided(binarization_wsr,_single_pc,n_subjects,n_regions,n_lambda)
    # 对稀疏引导处理后的数据进行转换
    sparse_guided_matrix = transform(_sparse_guided,n_subjects,n_regions,n_lambda)
    # 将稀疏引导矩阵和标签数据进行随机打乱，并返回
    data_all,label_all = shuffle(sparse_guided_matrix,label,random_state=0)
    return data_all,label_all
# data格式：n_subjects,n_lambda-1, n_regions, n_regions
