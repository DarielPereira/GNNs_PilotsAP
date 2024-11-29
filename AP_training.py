import os
import torch as th
import glob
from functionsGraphHandling import SampleBuffer, MyGraphDataset, SingleLayerGNN
import torch.utils.data as th_data
from torch_geometric.loader import DataLoader

# Stored training data
try:
    graphs = th.load(f'./AP_TrainingData/AP_training_Dataset.pt')
except:
    graphs = []

# Create the dataset and load the information in the buffers
dataset = MyGraphDataset(root='./', graphs=graphs)

# Create a list of buffers
buffers = []
# pattern that repeats in the file names
pattern = f'./AP_TRAININGDATA/newData/SE_Comb_MR_L_225_N_1_M_7_I_3_taup_150_NbrSamp_'
filename = f'{pattern}*.pkl'
# Get the list of files that match the pattern
matching_files = glob.glob(filename)
# Load the buffers from the files
for file in matching_files:
    buffer = SampleBuffer(batch_size=10)
    buffer.load(file)
    buffers.append(buffer)
    os.rename(file, file.replace('newData', 'inDataSet'))

if buffers.__len__() > 0:
    # Add the buffers to the dataset
    dataset.buffers2dataset(buffers, filepath='./AP_TrainingData')

# set a seed for reproducibility
th.manual_seed(0)

# Shuffle the dataset
dataset = dataset.shuffle()

# Split the dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = th_data.random_split(dataset, [train_size, test_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model, optimizer, and loss
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = SingleLayerGNN(in_channels=dataset.num_features, out_channels=7).to(device)
optimizer = th.optim.Adam(model.parameters(), lr=0.01)
loss_fn = th.nn.BCEWithLogitsLoss()

model.train()
average_loss = 0
for data in train_loader:
    data = data.to(device)
    optimizer.zero_grad()

    embedding = model(data.x, data.edge_index)

    # Compute loss with the graph's vector label
    loss = loss_fn(embedding, data.y.to(th.float))
    loss.backward()
    optimizer.step()
    average_loss += loss.item()/len(train_loader)

    print(f'Loss: {loss.item()}')


print('End of the script')