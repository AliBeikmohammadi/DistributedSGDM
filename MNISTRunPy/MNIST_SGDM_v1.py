import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os

def maincode(args):
    # Set hyperparameters
    seed_value = args.seed_number
    learning_rate = args.learning_rate #learning rate
    beta = args.beta
    l2 = args.L2 #weight_decay (L2 penalty)

    n_nodes= args.num_nodes
    num_epochs = args.num_epochs
    method=args.method #'none''clip_grad_value' 'clip_grad_norm' 'Top-K' 'norm'
    max_norm=args.max_norm #in case 'clip_grad_norm'
    K= args.top_k #in case 'Top-K'
    clip_value = args.clip_value #in case 'clip_grad_value'
    
    #NN architecture
    image_size =28*28
    num_hidden1 = 512
    num_hidden2 = 512
    num_output = 10

    # Set seed value for reproducibility
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    # Load the MNIST dataset
    train_dataset = MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = MNIST('./data', train=False, download=True, 
                         transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

    batch_size = int(len(train_dataset)/n_nodes)
    NUM_ACCUMULATION_STEPS = n_nodes

    filename='MNIST_'+'A_'+str(num_hidden1)+str(num_hidden2)+'_penalty_'+str(l2)+'_e_'+str(num_epochs)+         '_N_'+str(n_nodes)+'_SGDM'+'_lr_'+str(learning_rate)+'_beta_'+str(beta)
    if method=='none':
        filename=filename
    elif method=='clip_grad_value':
        filename+='_ClipValue_'+str(clip_value) 
    elif method=='clip_grad_norm':
        filename+='_ClipNorm_'+str(max_norm)
    elif method=='Top-K':
        filename+='_TopK_'+str(K)
    elif method=='norm':
        filename+='_Norm'
    filename+='_seed_'+str(seed_value)
    if beta==1:
        filename=filename.replace('SGDM','SGD')
        filename=filename.replace('_beta_1_','_')
        filename=filename.replace('_beta_1.0_','_')
    writer = SummaryWriter('runs/'+filename)
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # Define the model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(image_size, num_hidden1),
        nn.ReLU(),
        nn.Linear(num_hidden1, num_hidden2),
        nn.ReLU(),
        nn.Linear(num_hidden2, num_output)
    )
    print('Trainable params', sum(param.numel() for param in model.parameters() if param.requires_grad))

    # Define the loss function and optimizer
    momentum = 1-beta #momentum factor https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    tau= 1-beta #dampening for momentum https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=tau, weight_decay=l2, nesterov=False, maximize=False)

    # Train the model
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        # Initialize a variable to accumulate gradients
        total_grads = None

        # Train
        model.train()
        for idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward Pass
            outputs = model(inputs)
            # Compute Loss and Perform Back-propagation
            loss = criterion(outputs, labels)
            # Backward pass with retain_graph=True to keep the gradients
            loss.backward(retain_graph=True)

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()

            if method=='clip_grad_norm':
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)
            elif method=='clip_grad_value':
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
            elif method=='norm':
                # set max_norm to infinity for normalized gradients
                for param in model.parameters():  #Maybe just on weights not bias
                    param.grad /= torch.norm(param.grad, p=2)
            elif method=='Top-K':
                flatten_grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        flatten_grad.append(param.grad.view(-1))
                flatten_grad = torch.cat(flatten_grad)
                k= min(K, len(flatten_grad))
                topk, _ = torch.topk(torch.abs(flatten_grad), k)
                threshold = topk[-1]
                for param in model.parameters():
                    if param.grad is not None:
                        mask = torch.abs(param.grad) >= threshold
                        param.grad *= mask.type_as(param.grad)

            # Accumulate the gradients
            if total_grads is None:
                total_grads = [param.grad.clone() for param in model.parameters()]
            else:
                for i, param in enumerate(model.parameters()):
                    total_grads[i] += param.grad.clone()

            if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0): #all nodes has computed gradients on their part
                # Compute the average gradients
                for i in range(len(total_grads)):
                    total_grads[i] /= NUM_ACCUMULATION_STEPS
                # Update the model parameters using the average gradients
                for i, param in enumerate(model.parameters()):
                    param.grad = total_grads[i]
                # Update Optimizer
                optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for this epoch
        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc / len(train_dataset)
        test_loss = test_loss / len(test_dataset)
        test_acc = test_acc / len(test_dataset)
        
        # Write on Tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        # Append the results to the lists
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # Print the results for this epoch
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')
    
    # Save the results to a CSV file
    results_dict = {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_loss': test_loss_list,
        'test_acc':test_acc_list,
    }
    results_df = pd.DataFrame(results_dict)
    # Check whether the specified path exists or not
    isExist = os.path.exists('./MNIST_CSV/')
    if not isExist:
   # Create a new directory because it does not exist
       os.makedirs('./MNIST_CSV/')
    results_df.to_csv('./MNIST_CSV/'+filename+'.csv', index=True)
    # Close Tensorboard
    writer.flush()
    writer.close()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST')
    parser.add_argument('--seed_number', type=int, default=1,
                        help='seed number')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate for the optimizer')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta for the optimizer')
    parser.add_argument('--L2', type=float, default=1e-5,
                        help='weight_decay (L2 penalty)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--num_nodes', type=int, default=20,
                        help='number of nodes')
    parser.add_argument('--method', type=str, default='Top-K',
                        help='method: none, norm, clip_grad_norm, clip_grad_value, Top-K')
    parser.add_argument('--max_norm', type=float, default=1.0,
                    help='gradient norm clipping max value (necessary if method: clip_grad_norm)')
    parser.add_argument('--top_k', type=int, default=100,
                    help='number of top elements to keep in compressed gradient (necessary if method: Top-K)')
    parser.add_argument('--clip_value', type=float, default=10,
                    help='gradient clipping value (necessary if method: clip_grad_value)')
    
args = parser.parse_args()
# Train the network
maincode(args)
