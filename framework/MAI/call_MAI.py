import numpy as np
import torch
from torch import nn
import pandas as pd
import sys
import joblib
from openpyxl import Workbook
from torch.utils.data import Dataset, DataLoader
sys.path.append('./')

# Define mini-batch
class CustomDataset(Dataset):
    def __init__(self, inputs1, inputs2, labels):
        self.len=inputs1.shape[0]
        self.inputs1=torch.from_numpy(inputs1)
        self.inputs2=torch.from_numpy(inputs2)
        self.labels=torch.from_numpy(labels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.inputs1[index], self.inputs2[index], self.labels[index]


# Define CNN Neural Networks
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(1, 75, 3)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(75, 150, 2)
        self.linear = nn.Linear(150, 150)

    def forward(self, x):
        x=self.dropout(x)
        x = x.float()
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# Define CNN-LSTM Neural Networks
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(150, 200, 2, batch_first=True, dropout=0.3)
        self.linear1 = nn.Linear(200, 195)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 121)
        self.ReLU=nn.ReLU()

    def forward(self, x1, x2):
        batch_size, sequence = x1.shape[0], x1.shape[1]
        data_cnn_x1 = torch.zeros(batch_size, sequence, 150).to(x1.device)

        for i in range(sequence):
            data_cnn = x1[:, i, :]
            data_cnn = data_cnn.reshape(-1, 1, 11, 11)
            data_cnn = self.cnn(data_cnn)
            data_cnn = data_cnn.reshape(-1, 1, 150)
            data_cnn_x1[:, i, :] = data_cnn.squeeze(1)

        x, _ = self.lstm(data_cnn_x1)  # x1 is input, size (batch, seq_len, input_size) 对时序数据进行lstm

        data_dense = x[:, -1, :]  # 取最后一个时间点的结果
        data_dense = data_dense.unsqueeze(1)
        data_dense = self.linear1(data_dense)
        data_dense = self.ReLU(data_dense)
        data_dense = torch.cat([data_dense, x2], dim=2)
        data_dense = data_dense.float()
        data_dense = self.linear2(data_dense)
        data_dense = self.ReLU(data_dense)
        data_dense = self.linear3(data_dense)
        data_dense = self.ReLU(data_dense)
        data_dense = self.linear4(data_dense)
        return data_dense

# Define PELSTM Neural Networks
class PELSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(125, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 196)
        self.lstm = nn.LSTM(200, 10, 2, batch_first=True, dropout=0.3)
        self.linear4 = nn.Linear(10,1)
        self.ReLU=nn.ReLU()

    def forward(self, x1, x2):
        batch_size, sequence = x1.shape[0], x1.shape[1]
        data_x = torch.zeros(batch_size, sequence, 200).to(x1.device)
        
        input_x=torch.cat([x1, x2], dim=2)
        input_x=self.linear1(input_x)
        input_x = self.ReLU(input_x)
        input_x=self.linear2(input_x)
        input_x = self.ReLU(input_x)
        input_x=self.linear3(input_x)
        input_x= self.ReLU(input_x)
        data_x =torch.cat([input_x, x2], dim=2)

        x, _ = self.lstm(data_x)  # 对时序数据进行lstm
        x=x[:, -1, :]  # 取最后一个时间点的轴力比
        x=self.linear4(x)
        return x
    
# 数据损失
def data_loss(model, x1, x2):
    p = model(x1, x2)  # lstm计算得到的轴力比
    return p    

def calculate_rmse(matrix1, matrix2):
    # 计算差的平方
    squared_diff = np.square(matrix1 - matrix2)
    # 计算MSE
    mse = np.mean(squared_diff)
    # 计算RMSE
    rmse = np.sqrt(mse)
    return rmse

def calculate_r2(actual, predicted):
    # 计算总平方和（TSS）
    tss = np.sum((actual - np.mean(actual))**2)
    # 计算残差平方和（RSS）
    rss = np.sum((actual - predicted)**2)
    # 计算R^2
    r2 = 1 - rss / tss
    return r2


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_device = torch.cuda.current_device()
    TFPM_scaler_data = joblib.load(r"E:\qhh\overallperformance\MAI\TFPM_scaler_data.save")  # 导入归一化参数
    AFPM_T_scaler_data = joblib.load(r"E:\qhh\overallperformance\MAI\AFPM_T_scaler_data.save")  # 导入归一化参数
    AFPM_x_scaler_data = joblib.load(r"E:\qhh\overallperformance\MAI\AFPM_x_scaler_data.save")  # 导入归一化参数
    column_width=0.4

# ================================ 数据导入和处理
# 加载数据
    test_x_4T = np.array(
        pd.read_excel(r"E:\qhh\overallperformance\application\measuretemperature.xlsx", header=None))    
    test_x_x = np.array(
        pd.read_excel(r"E:\qhh\overallperformance\application\axialdeformation.xlsx", header=None))      

    FF = test_x_4T.shape[0]
    test_x_4T = test_x_4T.astype('float32')
    dummy_x2 = np.full((FF, 121), 500)
    for i in range(FF):
        dummy_x2[i, 5] = test_x_4T[i, 0]
        dummy_x2[i, 55] = test_x_4T[i, 1]
        dummy_x2[i, 65] = test_x_4T[i, 2]
        dummy_x2[i, 115] = test_x_4T[i, 3]
    dummy_x2 = dummy_x2.astype('float32')
    dummy_x2 = TFPM_scaler_data.transform(dummy_x2)
    for i in range(FF):
        test_x_4T[i, 0] = dummy_x2[i, 5]
        test_x_4T[i, 1] = dummy_x2[i, 55]
        test_x_4T[i, 2] = dummy_x2[i, 65]
        test_x_4T[i, 3] = dummy_x2[i, 115]
    data_b = np.full((FF, 1), column_width)
    test_x_4TB = np.concatenate([test_x_4T, data_b], axis=1)
    test_x_4TB = test_x_4TB[np.newaxis, :, :]

    total_batch_size = test_x_4TB.shape[0]
    test_x_121T = np.full((total_batch_size,96,121),0)  # 经过归一化后常温20度为0,初始的温度场
    
    # 取第二维的第一组数据，同时保持三维结构  
    test_x_x = AFPM_x_scaler_data.transform(test_x_x)
    test_x_x= test_x_x[np.newaxis, :, :]
    selected_data = test_x_x[:, 0:1, :]
    # 将选中的数据复制四次
    test_x_x_initial = np.tile(selected_data, (1, 4, 1))
    
#    Mydataset_test = CustomDataset(test_x1, test_x2, test_y)
#    test_loader = DataLoader(dataset=Mydataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0)
# ================================计算cnnlstm模型
    cnnlstm_model = CNNLSTM().to(device)
    cnnlstm_model=cnnlstm_model.float()
    cnnlstm_model.load_state_dict(
        torch.load(r"E:\qhh\temperature distribution\cnnlstm4.pt", map_location=device))
    cnnlstm_model = cnnlstm_model.eval()
    
    pelstm_model = PELSTM().to(device)
    pelstm_model=pelstm_model.float()
    pelstm_model.load_state_dict(
        torch.load(r"E:\qhh\axialforce\lstm_64batch.pt", map_location=device))
    pelstm_model = pelstm_model.eval()
    
    test_x1_tensor=torch.tensor(test_x_121T).float().to(device)
    test_x2_tensor=torch.tensor(test_x_4TB).float().to(device)
    test_x_x_tensor=torch.tensor(test_x_x_initial).float().to(device)
    test_x4_tensor=torch.tensor(test_x_x).float().to(device)
    seq_len = test_x2_tensor.shape[1]
    test_prediction_out=np.zeros((seq_len,1))
    for j in range(seq_len):
        test_x2_one_tensor=test_x2_tensor[:,j,:]  
        test_x2_one_tensor=test_x2_one_tensor.reshape(-1,1,5) 
        output = cnnlstm_model(test_x1_tensor, test_x2_one_tensor)
        test_x1_tensor = torch.cat([test_x1_tensor[:, 1:seq_len, :], output.detach()], dim=1)
        
        test_x_TFPM = test_x1_tensor[:, -4:, :].cpu().numpy()  # 两个模块的归一化不同，要转换一下
        test_x_T_tensor=np.zeros_like(test_x_TFPM)
        num_slices=test_x_T_tensor.shape[0]
        for i in range(num_slices):
            slice_2d = test_x_TFPM[i, :, :]  # 取出一个二维切片
            data_x1_real_slice = TFPM_scaler_data.inverse_transform(slice_2d)    # 温度场真实数据121个温度每条
            slice_normalized = AFPM_T_scaler_data.transform(data_x1_real_slice)  # 对二维切片归一化
            test_x_T_tensor[i,:,:]=slice_normalized    
        test_x_T_tensor=torch.tensor(test_x_T_tensor).to(device='cuda')
        test_x_x_tensor = torch.cat([test_x_x_tensor[:, -3:, :], test_x4_tensor[:, j:j+1, :]], dim=1)
        test_prediction = data_loss(pelstm_model, test_x_T_tensor, test_x_x_tensor)
        test_prediction_out[j,0] = test_prediction.detach().cpu()
    
    wb = Workbook()
    ws = wb.active
    for i in range(FF):
        ws.cell(row=i+1, column=1).value = test_prediction_out[i,0]

    wb.save('estimation.xlsx')
