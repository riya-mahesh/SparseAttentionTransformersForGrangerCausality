"""
This file must contain a function called my_method that triggers all the steps 
required in order to obtain

 *val_matrix: mandatory, (N, N) matrix of scores for links
 *p_matrix: optional, (N, N) matrix of p-values for links; if not available, 
            None must be returned
 *lag_matrix: optional, (N, N) matrix of time lags for links; if not available, 
              None must be returned

Zip this file (together with other necessary files if you have further handmade 
packages) to upload as a code.zip. You do NOT need to upload files for packages 
that can be imported via pip or conda repositories. Once you upload your code, 
we are able to validate results including runtime estimates on the same machine.
These results are then marked as "Validated" and users can use filters to only 
show validated results.

Shown here is a vector-autoregressive model estimator as a simple method.
"""

#Attention
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np


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


def generate_mask(size):
  mask = torch.tril(torch.ones(size, size))
  mask[mask == 0] = float('-inf')
  mask[mask == 1] = 0
  return mask


class Attention(nn.Module):
  def __init__(self, seq_length, d_model): #seq_length = 4, d_model=3, k= no.of elements to be picked
    super(Attention, self).__init__()
    self.d_model = d_model
    self.seq_length = seq_length

    # Positional embedding of query vector
    self.pos1 = PositionalEmbedding(seq_length,d_model).double()
    self.pos2 = PositionalEmbedding(d_model, self.seq_length).double()

    # Linear layers for temporal attention computation
    self.W_q1 = nn.Linear(d_model, d_model).double()
    self.W_k1 = nn.Linear(d_model, d_model).double()

    # Linear layers for spatial attention computation
    self.W_q2 = nn.Linear(self.seq_length, self.seq_length).double()
    self.W_k2 = nn.Linear(self.seq_length, self.seq_length).double()
    self.W_v2 = nn.Linear(self.seq_length, self.seq_length).double()
    self.W_o2 = nn.Linear(self.seq_length, 1).double()


  def forward(self,q, spatial_mask = None):
    q_enc1 = self.pos1.forward(q,None)

    # Temporal attention
    query1 = self.W_q1(q_enc1)
    key1 = self.W_k1(q_enc1)
    # Generating mask to prevent future tokens from attending
    temporal_mask = generate_mask(self.seq_length)
    attn_scores = (torch.matmul(query1, key1.transpose(-2, -1))+temporal_mask) / math.sqrt(self.d_model)
    attn_probs1 = torch.softmax(attn_scores, dim=-1)
    attn_probs1 = attn_probs1.squeeze()
    column_sums = torch.sum(attn_probs1, dim=0)
    #print(attn_probs1)
    #print(column_sums)

    column_sums = column_sums/torch.sum(column_sums)

    column_sums=column_sums.view(-1, 1)

    # Passing the elements in the query as a weighted average of the column sums of temporal attention layer
    query = q*column_sums

    query = query.t()

    # Spatial attention
    q_enc2 = self.pos2.forward(query,None)
    q2 = self.W_q2(q_enc2)
    k2 = self.W_k2(q_enc2)
    v2 = self.W_v2(q_enc2)
    if spatial_mask == None:
      attn_scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.seq_length)
    else:
      attn_scores = (torch.matmul(q2, k2.transpose(-2, -1))+ spatial_mask) / math.sqrt(self.seq_length)

    attn_probs2 = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs2, v2)
    attn_output = self.W_o2(output)
    attn_output = attn_output.squeeze()
    attn_probs2 = attn_probs2.squeeze()
    #print(attn_output,attn_probs2)
    return attn_output, attn_probs2

class Transformer(nn.Module):
  def __init__(self, seq_length, d_model, lr, desired_loss):
    super(Transformer, self).__init__()
    self.d_model = d_model
    self.seq_length = seq_length
    self.attn = Attention(seq_length, d_model)
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(self.attn.parameters(), lr)
    self.desired_loss = desired_loss

  def forward(self,q,output_vector):
    while True:
      self.optimizer.zero_grad()
      # Forward pass
      attn_output, scores = self.attn.forward(q)

      # Compute the loss between attn_output and the required output
      loss = self.criterion(attn_output, output_vector)
      # Backward pass
      loss.backward()  # Calculate gradients
      self.optimizer.step()  # Update weights

      #early_stopping = nn.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

      if loss.item() < self.desired_loss:
        break

    return attn_output, scores, self.attn

    
# Your method must be called 'my_method'
# Describe all parameters (except for 'data') in the method registration on CauseMe
def my_method(data, maxlags, window_size):

    X_train,y_train=shape_data(data,maxlags,window_size,True)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)

    seq_length = X_train.shape[1]
    d_model = X_train.shape[2]

    error_unrestricted = []
    error_restricted = {} #dictionary with keys as variable index being masked and values as corresponding errors
    for i in range(d_model):
        error_restricted[i] = []

    for m in range(X_train.shape[0]):
        output = y_train[m].squeeze()
        #print(output.shape)
        trans = Transformer(seq_length=seq_length, d_model=d_model, lr=0.01, desired_loss=0.01)
        attn_output, scores, model = trans.forward(X_train[m],output)
        error = abs(output-attn_output)
        error_unrestricted.append(error)
        model.eval()

        with torch.no_grad():  # Disable gradient tracking for inference
            for i in range(d_model):
                spatial_mask = torch.zeros(d_model, d_model)
                for j in range(d_model):
                    spatial_mask[j][i] = float('-inf')
                attn_output, scores = model.forward(X_train[m], spatial_mask)
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

    causal_matrix = np.array(causal_matrix)
    max_element = np.max(causal_matrix)



    val_matrix = causal_matrix/max_element



    return val_matrix


if __name__ == '__main__':

    # Simple example: Generate some random data
    data = np.random.randn(100000, 3)

    # Create a causal link 0 --> 1 at lag 2
    data[2:, 1] -= 0.5*data[:-2, 0]

    # Estimate VAR model
    vals, pvals, lags = my_method(data, maxlags=5, window_size=0)

    # Score is just absolute coefficient value, significant p-value is at entry 
    # (0, 1) and corresponding lag is 2
    print(vals.round(2))
    print(pvals.round(3))
    print(pvals < 0.0001)
    print(lags)
