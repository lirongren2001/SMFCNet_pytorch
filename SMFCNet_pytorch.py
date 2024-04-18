import torch
import torch.nn as nn

SBN_num = 10
ROI_num = 90
batch_size = 20
channels_total = 230

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
    
class SMFC_Net(nn.Module):
    def __init__(self, args):
        super(SMFC_Net, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                # nn.Conv2d(1, 64, kernel_size=(1, ROI_num), padding='valid'),
                nn.Conv2d(1, 64, kernel_size=(1, args.num_nodes), padding='valid'),

                nn.ReLU(),
                # nn.Conv2d(64, 32, kernel_size=(ROI_num, 1), padding='valid'),
                nn.Conv2d(64, 32, kernel_size=(args.num_nodes, 1), padding='valid'),

                nn.ReLU(),
                Reshape(1, 32)
            ) for _ in range(args.SBN_num)
        ])

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(32 * args.SBN_num, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print('x.x.shape',x.shape)
        x = x.view(x.shape[0]//x.shape[1], x.shape[1], x.shape[1])
        # x = x.view(x.shape[0]//230, 230, x.shape[1])
        print('x.x.shape',x.shape)



        # x = [conv(x_i.unsqueeze(1)) for conv, x_i in zip(self.conv_layers, x)]
        # print('x.unsqueeze(1)',x.unsqueeze(0).unsqueeze(0).shape)
        # print('x.x.shape',x.shape)

        x = [conv(x.unsqueeze(1)) for conv, x_i in zip(self.conv_layers, x)]

        x = torch.cat(x, dim=1)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Create an instance of the model
# model = SMFC_Net()
