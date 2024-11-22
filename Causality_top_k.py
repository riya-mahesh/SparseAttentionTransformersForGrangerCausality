import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

file_path = "N-3 T-150/linear-VAR_N-3_T-150_0001.txt"
data = []
# Read data from the file
with open(file_path, 'r') as file:
  for line in file:
#Split each line by spaces, convert values to float, and append to the data list
    values = [float(x) for x in line.split()]
    data.append(values)
data=np.array(data)


# Shaping the data into batches of consecutive lags windows
def shape_data(arr, l_s, w, train):

  # l_s : The number of previous data samples to be sent to predict next value
  # w : Length of the window around lag l_s
  # arr is an array of shape [timesteps,input dimensions]

  X= []
  for i in range(len(arr) - l_s - w):
    X.append(arr[i:i + l_s + w])
  X = np.array(X)

  # data consists of sets of array entires of the dimension [len(arr)-l_s-n_pred,l_s + n_pred,input dimensions ]
  y = []
  for i in range(l_s + w,len(arr)):
    y.append(arr[i:i+1])
  y = np.array(y)

  assert len(X.shape) == 3
  assert len(y.shape) == 3

  # Defining training and test data set
  X_train = []
  y_train = []
  X_test = []
  y_test = []
  # Set train True while passing training dataset
  if train:
      X_train = list(X)
      X_train=np.array(X_train)
      y_train=np.array(y)
      return X_train,y_train


class PositionalEmbedding(nn.Module):
  def __init__(self, seq_length, d_model):
      super(PositionalEmbedding, self).__init__()
      self.d_model = d_model
      self.seq_length = seq_length
      self.position_embeddings = nn.Embedding(seq_length, d_model)

  def forward(self, input_ids, position_ids=None):
      seq_length = input_ids.size(0)
      if position_ids is None:
        position_ids = torch.arange(seq_length).unsqueeze(0)
      position_embeddings = self.position_embeddings(position_ids)
      embeddings = input_ids + position_embeddings
      return embeddings


def generate_temporal_mask(size):
  mask = torch.tril(torch.ones(size, size))
  mask[mask == 0] = float('-inf')
  mask[mask == 1] = 0
  return mask


class Temporal_Attention(nn.Module):
    def __init__(self, seq_length, d_model): #seq_length = 4, d_model=3, k= no.of elements to be picked
      super(Temporal_Attention, self).__init__()
      self.d_model = d_model
      self.seq_length = seq_length

      # Positional embedding of query vector
      self.pos1 = PositionalEmbedding(seq_length,d_model).double()


      # Linear layers for temporal attention computation
      self.W_q1 = nn.Linear(d_model, d_model).double()
      self.W_k1 = nn.Linear(d_model, d_model).double()
      self.W_v1 = nn.Linear(d_model, d_model).double()
      self.W_o1 = nn.Linear(self.seq_length, 1).double()


    def forward(self,q):
      q_enc1 = self.pos1.forward(q,None) #4*3
      q1 = self.W_q1(q_enc1) #4*3
      k1 = self.W_k1(q_enc1) #4*3
      v1 = self.W_v1(q_enc1) #4*3

      mask = generate_temporal_mask(self.seq_length)
      attn_scores = (torch.matmul(q1, k1.transpose(-2, -1)) + mask) / math.sqrt(self.d_model)
      attn_probs1 = torch.softmax(attn_scores, dim=-1)
      output = torch.matmul(attn_probs1, v1) #4*3
      output = output.squeeze()
      output2 = output.t()
      attn_output = self.W_o1(output2)
      attn_output = attn_output.squeeze()
      attn_probs1 = attn_probs1.squeeze()
      return attn_output, attn_probs1


class Temporal_Transformer(nn.Module):
    def __init__(self, seq_length, d_model):
        super(Temporal_Transformer, self).__init__()
        self.attn = Temporal_Attention(seq_length, d_model)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.attn.parameters(), lr=0.01)

    def train_model(self, q, output_vector, desired_loss):
        while(True):
            self.optimizer.zero_grad()
            # Forward pass
            attn_output, scores = self.attn.forward(q)
            # Compute the loss between attn_output and the required output
            loss = self.criterion(attn_output, output_vector)
            # Backward pass
            loss.backward()  # Calculate gradients
            self.optimizer.step()  # Update weights

            if loss.item() < desired_loss:
              break

        return attn_output, scores    

X_train,y_train=shape_data(data,10,2,True)
print(X_train.shape)
print(y_train.shape)
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)


temporal_transformer_model = Temporal_Transformer(seq_length=12, d_model=3)
# Train the model
output = y_train[1].squeeze()
attn_output, scores = temporal_transformer_model.train_model(X_train[1], output, desired_loss=0.0001)


def query_top_k(query, scores, k):
  column_sums = torch.sum(scores, dim=0)
  _, indices = torch.sort(column_sums, descending=True)
  top_k_indices = indices[:k]
  print(top_k_indices)
  return query[top_k_indices]


processed_query = query_top_k(X_train[0],scores, 5).t()
print(processed_query.size())


class Spatial_Attention(nn.Module):
  def __init__(self, k, d_model): #seq_length = 149, d_model=3, k= no.of elements to be picked
    super(Spatial_Attention, self).__init__()
    self.k = k # Pick top k elements
    self.d_model = d_model


    # Positional embedding of query vector
    self.pos2 = PositionalEmbedding(self.d_model, self.k).double()

    # Linear layers for spatial attention computation
    self.W_q2 = nn.Linear(self.k, self.k).double()
    self.W_k2 = nn.Linear(self.k, self.k).double()
    self.W_v2 = nn.Linear(self.k, self.k).double()
    self.W_o2 = nn.Linear(self.k, 1).double()

  def forward(self,query,mask=None):

    # Spatial attention
    q_enc2 = self.pos2.forward(query,None)
    q2 = self.W_q2(q_enc2)
    k2 = self.W_k2(q_enc2)
    v2 = self.W_v2(q_enc2)
    if mask == None:
      attn_scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.k)
    else:
      attn_scores = (torch.matmul(q2, k2.transpose(-2, -1))+ mask) / math.sqrt(self.k)

    attn_probs2 = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs2, v2)
    attn_output = self.W_o2(output)
    attn_output = attn_output.squeeze()
    attn_probs2 = attn_probs2.squeeze()
    # print(attn_output,attn_probs2)
    return attn_output, attn_probs2

class Spatial_Transformer(nn.Module):
    def __init__(self, k, d_model, mask=None):
        super(Spatial_Transformer, self).__init__()
        self.attn = Spatial_Attention(k, d_model)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.attn.parameters(), lr=0.01)
        self.mask = mask

    def train_model(self, q, output_vector, desired_loss):
        while(True):
            self.optimizer.zero_grad()
            # Forward pass
            attn_output, scores = self.attn.forward(q, self.mask)
            # Compute the loss between attn_output and the required output
            loss = self.criterion(attn_output, output_vector)
            # Backward pass
            loss.backward()  # Calculate gradients
            self.optimizer.step()  # Update weights

            if loss.item() < desired_loss:
              break

        return attn_output, scores, self.attn


spatial_transformer_model = Spatial_Transformer(k=4, d_model=3)
# Train the model
attn_output, scores, model = spatial_transformer_model.train_model(processed_query, output, desired_loss=0.0001)


X_train,y_train=shape_data(data,10,1,True)
print(X_train.shape)
print(y_train.shape)
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

seq_length = X_train.shape[1]
d_model = X_train.shape[2]
k = 5 #Top how many indices 

error_unrestricted = []
error_restricted = {} #dictionary with keys as variable index being masked and values as corresponding errors
for i in range(d_model):
    error_restricted[i] = []

for m in range(X_train.shape[0]):
    
    output = y_train[m].squeeze()
    print(output)
    temporal_transformer_model = Temporal_Transformer(seq_length=seq_length, d_model=d_model)
    attn_output, scores = temporal_transformer_model.train_model(X_train[m], output, desired_loss=0.01)
    print(attn_output)
    processed_query = query_top_k(X_train[m],scores, k).t()
    print(processed_query)
    
    spatial_transformer_model = Spatial_Transformer(k=k, d_model=d_model)
    # Train the model
    attn_output, scores, model = spatial_transformer_model.train_model(processed_query, output, desired_loss=0.01)
    print(attn_output)
    error = abs(output-attn_output)
    error_unrestricted.append(error)
    
    model.eval()

    with torch.no_grad():  # Disable gradient tracking for inference
      for i in range(d_model):
        spatial_mask = torch.zeros(3, 3)
        for j in range(d_model):
          spatial_mask[j][i] = float('-inf')
        attn_output, scores = model.forward(processed_query, spatial_mask)
        error_restricted[i].append(abs(attn_output-output))
      
# Granger causality index calculation
#Unrestricted model standard deviation
temp1 = torch.stack(error_unrestricted)
std_unrestricted = torch.std(temp1, dim=0)

#Restricted model standard deviation
std_restricted=[]
for i in range(d_model):
    temp2 = torch.stack(error_restricted[i])
    std_restricted.append(torch.std(temp2, dim=0))

causal_matrix = []
for i in range(d_model):
    temp = []
    for j in range(d_model):
        temp.append(2*math.log(std_restricted[i][j]/std_unrestricted[i]))
    causal_matrix.append(temp)