import torch as th
import glob

from functionsGraphHandling import SampleBuffer, MyGraphDataset

# Create a list of buffers
buffers = []
# pattern that repeats in the file names
pattern = f'./AP_TRAININGDATA/SE_Comb_MR_L_225_N_1_M_7_I_3_taup_150_NbrSamp_'
filename = f'{pattern}*.pkl'
# Get the list of files that match the pattern
matching_files = glob.glob(filename)
# Load the buffers from the files
for file in matching_files:
    buffer = SampleBuffer(batch_size=10)
    buffer.load(file)
    buffers.append(buffer)

# Create the dataset and load the information in the buffers
dataset = MyGraphDataset(root='./')
dataset.buffers2dataset(buffers)

print('End of the script')