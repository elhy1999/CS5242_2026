import time
import torch
import torch.nn as nn

################## Initialization ##################
my_tensor = torch.Tensor([1, 1]) # float tensor
my_long_tensor = my_tensor.long() # LongTensor
input_dim = 2; output_dim = 3; linear_layer = nn.Linear(input_dim, output_dim, bias=True) # Matrix of linear_layer is (output_dim x input_dim)
softmax_layer = nn.Softmax(dim=0)
torch.softmax(my_tensor, dim=0)

class OneLayerNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(OneLayerNet , self).__init__()
        self.linear_layer = nn.Linear( input_size, output_size , bias=False)
        
    def forward(self, x):
        y = self.linear_layer(x)
        prob = torch.softmax(y, dim=1)
        return prob

class ThreeLayerNet(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2,  output_size):
        super(ThreeLayerNet , self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.layer3 = nn.Linear(hidden_size2, output_size, bias=False)
        
    def forward(self, x):
        
        y       = self.layer1(x)
        y_hat   = torch.relu(y)
        z       = self.layer2(y_hat)
        z_hat   = torch.relu(z)
        scores  = self.layer3(z_hat)
        
        return scores

net = OneLayerNet(...)

######### To check weight and bias values #########
linear_layer.weight
linear_layer.bias

######### Setting weights and bias values #########
with torch.no_grad():
    linear_layer.weight[0, 0] = 1
    linear_layer.weight[0, 1] = 2
    ...


############### Cross Entropy Loss ################
criterion = nn.CrossEntropyLoss()
labels = torch.LongTensor([0, 1, 1])
scores = torch.Tensor([
        [1, -1],
        [-1, 1],
        [-1, 1],
    ]
)
criterion(scores, labels)

################## Finalized Training loop ##################
N = 60000; n_dim = 3; w = h = 28; train_data = test_data = torch.rand([N, n_dim, w, h]); train_label = test_label = torch.rand([N, 1]).long()

criterion = nn.NLLLoss() # Used when you already computed log-probabilties (e.g. with log_softmax)
criterion = nn.CrossEntropyLoss() # Used on raw logits
# For definition, see slide 165 of combined lecture notes
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
batch_size = 200
num_epochs = 5000
for epoch in range(num_epochs):
    running_loss = 0
    running_error = 0
    num_batches = 0
    shuffled_indices = torch.randperm(N)
    for count in range(0, N, batch_size):
        # Set dL/dU, dL/dV, dL/dW to be filled with zeros
        optimizer.zero_grad()
        
        # create a minibatch
        # indices = torch.LongTensor(batch_size).random_(0, N)
        indices = shuffled_indices[count: count + batch_size]
        minibatch_data = train_data[indices]
        minibatch_label = train_label[indices]
        
        # reshape the minibatch
        inputs = minibatch_data.view(batch_size, 784)
        
        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net  
        minibatch_scores = net(inputs) 
        
        # Compute the average of the losses of the data points in the minibatch
        loss = criterion(minibatch_scores, minibatch_label) 
        
        # backward pass to compute dL/dU, dL/dV and dL/dW    
        loss.backward()
        
        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()

        # compute and accumulate stats
        running_loss += loss.detach().item()
        error = utils.get_error(minibatch_scores.detach(), minibatch_label) # Definition pasted below
        running_error += error.item()
        num_batches += 1
    
    # compute stats for the full training set
    total_loss = running_loss / num_batches
    total_error = running_error / num_batches
    print('epoch=', epoch, '\t loss=', total_loss, '\t error=', total_error*100, 'percent')

################## Finalized Training loop (with learning rate strategy) ##################
start = time.time()
lr = 0.05 # initial learning rate
for epoch in range(num_epochs):
    # learning rate strategy : divide the learning rate by 1.5 every 10 epochs
    if epoch % 10 == 0 and epoch > 10: 
        lr /= 1.5
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.   
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # Tracking metrics
    running_loss = 0
    running_error = 0
    num_batches = 0
    shuffled_indices = torch.randperm(N)
    for count in range(0, N, batch_size):
        # Set dL/dU, dL/dV, dL/dW to be filled with zeros
        optimizer.zero_grad()

        # create a minibatch
        # indices = torch.LongTensor(batch_size).random_(0, N)
        indices = shuffled_indices[count: count + batch_size]
        minibatch_data = train_data[indices]
        minibatch_label = train_label[indices]

        # reshape the minibatch
        inputs = minibatch_data.view(batch_size, 784) # Can also be (-1, 784) to let Pytorch automatically infer the batch size

        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net  
        scores=net(inputs) 

        # Compute the average of the losses of the data points in the minibatch
        # Note: The shape of scores has to be (batch_size, num_classes) 
        # and the shape of minibatch_label has to be (batch_size,) for this to work. 
        # If not, you will get an error.
        loss = criterion(scores, minibatch_label) 

        # backward pass to compute dL/dU, dL/dV and dL/dW    
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        
        # compute and accumulate stats
        running_loss += loss.detach().item()
        error = utils.get_error( scores.detach() , minibatch_label)
        running_error += error.item()
        num_batches+=1
    
    # once the epoch is finished we divide the "running quantities"
    # by the number of batches
    total_loss = running_loss / num_batches
    total_error = running_error / num_batches
    elapsed_time = time.time() - start
    
    # every 10 epoch we display the stats 
    # and compute the error rate on the test set  
    if epoch % 10 == 0:
        print(' ')
        print('epoch=',epoch, ' time=', elapsed_time,
              ' loss=', total_loss , ' error=', total_error*100 ,'percent lr=', lr)
        eval_on_test_set()
               

################## Evaluation on Test Set ##################
running_error = 0
num_batches = 0
for i in range(0, 10000, batch_size):

    # extract the minibatch
    minibatch_data = test_data[i: i + batch_size]
    minibatch_label = test_label[i: i + batch_size]

    # reshape the minibatch
    inputs = minibatch_data.view(batch_size, 784)

    # feed it to the network
    scores = net(inputs)

    # compute the error made on this batch
    error = utils.get_error(scores, minibatch_label)
    
    # add it to the running error
    running_error += error.item()

    num_batches+=1

# compute error rate on the full test set
total_error = running_error/num_batches
print( 'error rate on test set =', total_error*100 ,'percent')


################### Misc. ###################
def get_error(scores, labels):
    """
    Returns the classification error. Not to be used for regression errors
    """
    bs = scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = predicted_labels == labels
    num_matches = indicator.sum()

    return 1 - num_matches.float() / bs

def eval_on_test_set():
    running_error = 0
    num_batches = 0
    for i in range(0, N, batch_size):
        minibatch_data = test_data[i: i + batch_size]
        minibatch_label = test_label[i: i + batch_size]
        inputs = minibatch_data.view(batch_size, 784)
        scores = net(inputs)
        error = utils.get_error(scores, minibatch_label)
        running_error += error.item()
        num_batches+=1
    total_error = running_error/num_batches
    print('test error =', total_error*100 ,'percent')

################### PYP ###################

def my_softmax(x):
    exp_x = torch.exp(x)
    result = exp_x/ torch.sum(exp_x, dim = 1, keepdim=True)
    # Compute softmax
    return result


def my_nll_loss(log_probs, targets):
    """
    log_probs: Tensor of shape (N, C) containing log-probabilities for each class
    targets: Tensor of shape (N,) containing the indices of the correct class for each sample
    Returns:
    loss: Scalar tensor representing the average negative log-likelihood loss
    """
    N = log_probs.size(0)

    # Select the log-prob of the correct class for each sample
    correct_log_probs = log_probs[torch.arange(N), targets]

    # Negative mean
    loss = -correct_log_probs.mean()
    
    return loss

def my_cross_entropy(x, y):
    """
    x: Tensor of shape (N, C) containing the raw scores (logits) for each class
    y: Tensor of shape (N, C) containing the one-hot encoded true labels
    Returns:
    loss: Scalar tensor representing the average cross-entropy loss
    """
    if y.dim() == 1: # If y is not one-hot encoded (e.g. [0, 1, 0, 0, 1, 2]), convert it to one-hot encoding
        y = torch.nn.functional.one_hot(y, num_classes=x.size(1))
    log_probs = torch.log(my_softmax(x))
    return my_nll_loss(log_probs, y.argmax(dim=1))

class MyMLP:
    """
    A simple MLP with one hidden layer using manual matrix multiplication
    and manual sigmoid, without torch.nn.Linear or torch.sigmoid.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Weights and biases are randomly initialized with :
        #  1) Hidden layer: (input_dim  -> hidden_dim)
        #  2) Output layer: (hidden_dim -> output_dim)
        self.W1 = torch.randn(input_dim, hidden_dim, requires_grad=False)
        self.b1 = torch.randn(hidden_dim, requires_grad=False)
        self.W2 = torch.randn(hidden_dim, output_dim, requires_grad=False)
        self.b2 = torch.randn(output_dim, requires_grad=False)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def relu(self, x):
        return torch.maximum(torch.zeros_like(x), x)

    def forward(self, x):
        """
        Forward pass through:
            hidden = sigmoid(W1 * x + b1)
            output = W2 * hidden + b2
        """
        print(self.W1.shape) # (input_dim, hidden_dim)
        print(x.shape) # (N, input_dim)
        h = x @ self.W1 + self.b1
        h = self.sigmoid(h)
        out = h @ self.W2 + self.b2
        return out