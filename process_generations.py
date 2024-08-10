import torch
from sklearn.metrics import classification_report
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device, torch.cuda.is_available())


def train_net(net, data_classes, epochs, aug_name=None, participant='sttr001'):
    batch_size = 64
    loss_list = []
    i = 0
    for things in data_classes:
        pathsfront = './data_folder/'+ data_classes[i] + '/' + participant + '/'
        if(i == 0):
            data_tens = torch.load(pathsfront+data_classes[i] + '_train.pt')
            data_tens = data_tens.to(device)
            data_tens[torch.randint(data_tens.shape[0], (int(data_tens.shape[0]*0.1),)),:,:,:] 
            y_list = torch.ones((data_tens.shape[0]))*i
            
            test_tens = torch.load(pathsfront+data_classes[i] + '_val.pt')
            test_tens = test_tens.to(device)
            y_test = torch.ones((test_tens.shape[0]))*i

            if(aug_name == data_classes[i]):
                print("Loading from: ", pathsfront+'aug_analysis.pt')
                aug_tens = torch.load(pathsfront+'aug_analysis.pt')
                data_tens = torch.cat((data_tens, aug_tens)) 
                y_list = torch.ones((data_tens.shape[0]))*i
        else:
            new_tens = torch.load(pathsfront+data_classes[i] + '_train.pt')
            new_tens = new_tens.to(device)
            new_tens[torch.randint(new_tens.shape[0], (int(new_tens.shape[0]*0.1),)),:,:,:] 
            data_tens = torch.cat((data_tens, new_tens))
            
            y_new = torch.ones((new_tens.shape[0]))*i
            y_list = torch.cat((y_list, y_new))   
            
            if(aug_name == data_classes[i]):
                print("Loading from: ", pathsfront+'aug_analysis.pt')
                aug_tens = torch.load(pathsfront+'aug_analysis.pt')
                data_tens = torch.cat((data_tens, aug_tens))
                y_new = torch.ones((aug_tens.shape[0]))*i
                y_list = torch.cat((y_list, y_new))  
            
            new_test = torch.load(pathsfront+data_classes[i] + '_val.pt')
            new_test = new_test.to(device) 
            test_tens = torch.cat((test_tens, new_test))
            
            y_new_test = torch.ones((new_test.shape[0]))*i
            y_test = torch.cat((y_test, y_new_test)) 
            
        i += 1        
    y_list = y_list.type(torch.LongTensor)
    y_list = y_list.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    
    i = 0
    while i < epochs:        
        optim.zero_grad()
        indices = torch.randint(data_tens.shape[0], (batch_size,))
        sample = data_tens[indices] 
        samp_y = y_list[indices]
        out = net(sample)
        loss = loss_fn(out,samp_y)
        
        
        loss.backward()
        
        np_loss = loss.detach().cpu().numpy()
        
        if((i % 500 == 0) & 0):
            print("Loss at step " + str(i) + " is: ", np_loss)
            print(out)
        loss_list.append(np_loss)
        
        optim.step()
        
        i += 1
    accuracy = test_accuracy(net, test_tens, y_test)
    return(accuracy) #avf f1

def test_accuracy(model, X, y):
    output = model(X)
    _, predictions = torch.max(output, dim = 1)
    #print("Predictions are:", predictions)
    #print(len(predictions))
    #print("y vals are:", y)
    #print(len(y))
    cr = classification_report(y, predictions.cpu().numpy(), output_dict=True)
    cr_print = classification_report(y, predictions.cpu().numpy())
    cr = cr['macro avg']['f1-score']
    print(cr_print)
    return(cr)

class activity_recognizer(nn.Module):
    def __init__(self, out_size):
        super(activity_recognizer, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.out_size = out_size
        self.conv_layer = nn.Sequential(              
            nn.BatchNorm2d(4),     
            nn.Conv2d(4,4,(23,30),stride=(4,25),padding=(5),padding_mode='zeros'),
            nn.Dropout2d(p=0.35),   
            nn.BatchNorm2d(4),         
            nn.SELU(),
            nn.Conv2d(4,8,(1,2),stride=(2),padding=(0),padding_mode='zeros'),     
            nn.BatchNorm2d(8),              
            nn.SELU(),
            nn.Conv2d(8,32,(2,2),stride=(1),padding=(0),padding_mode='zeros'),
            nn.Flatten(start_dim=1),
            )
        self.lin_layer = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,self.out_size),
            )
    def forward(self, input):
        out = self.conv_layer(input)
        out = self.lin_layer(out)
        out = self.sm(out)
        return(out)

def generate_report_pred(name, participant):
    test_classes = ["errands", "exercise", "work"]
    epochs = 3000
    model = activity_recognizer(len(test_classes)).to(device)
    f1 = train_net(model, test_classes, epochs, name, participant)
    return(float(f1))



