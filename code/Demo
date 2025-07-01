import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

# Global parameters
K = 64
CP = K // 4
P = 64
allCarriers = np.arange(K)
pilotCarriers = allCarriers[::K//P]  # Pilots is every (K/P)th carrier
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2
payloadBits_per_OFDM = K * mu  #
SNRdb = 20  
Clipping_Flag = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# QPSK
mapping_table = {
    (0,0) : -1-1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : 1+1j,
}

def Modulation(bits):
    """QAM Modulation"""
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)

def OFDM_symbol(Data, pilot_flag):
    """OFDM_symbol"""
    symbol = np.zeros(K, dtype=complex)
    symbol[pilotCarriers] = pilotValue
    symbol[dataCarriers] = Data
    return symbol

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])

def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 
                                   1j * np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP + K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def ofdm_simulate(codeword, channelResponse, SNRdb):
    """ofdm_simulate"""
    # allCarriers
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue  
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam  
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_codeword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_codeword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    
    return np.concatenate([
        np.concatenate([np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)]),
        np.concatenate([np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)])
    ]), abs(channelResponse)

# Pilot
Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

pilotValue = Modulation(bits)

# channel data
try:
    channel_train = np.load('channel_train.npy')
    train_size = channel_train.shape[0]
    channel_test = np.load('channel_test.npy')
    test_size = channel_test.shape[0]
    print(f"Loaded channel data: train_size={train_size}, test_size={test_size}")
except FileNotFoundError:
    print("Channel data files not found.")

class OFDMDataset(Dataset):
    def __init__(self, channel_data, SNRdb=20, dataset_size=1000):
        self.channel_data = channel_data
        self.SNRdb = SNRdb
        self.dataset_size = dataset_size
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # channel
        channel_idx = np.random.randint(0, len(self.channel_data))
        H = self.channel_data[channel_idx]
        
        # random create bitdata
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        
        # OFDM
        signal_output, channel_abs = ofdm_simulate(bits, H, self.SNRdb)
        
        # input=signal_output, channel_abs(abs is additional)
        input_sample = torch.FloatTensor(np.concatenate([signal_output, channel_abs]))
        # label16-32
        label = torch.FloatTensor(bits[16:32])
        
        return input_sample, label

class OFDMNet(nn.Module):
    """OFDM DNN"""
    def __init__(self, input_size, n_hidden_1=500, n_hidden_2=250, n_hidden_3=120, n_output=16):
        super(OFDMNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc_out = nn.Linear(n_hidden_3, n_output)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc_out(x))
        return x

def bit_error_rate(y_pred, y_true):
    pred_sign = torch.sign(y_pred - 0.5)
    true_sign = torch.sign(y_true - 0.5)
    correct = (pred_sign == true_sign).float()
    accuracy = torch.mean(torch.mean(correct, dim=1))
    ber = 1 - accuracy
    return ber

def train_model():
   
    train_dataset = OFDMDataset(channel_train, SNRdb=SNRdb, dataset_size=50000)
    
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=0)
    
    input_size = 256 + len(channel_train[0])  
    model = OFDMNet(input_size).to(device)
    
    criterion = nn.MSELoss()  
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)  
    
    training_epochs = 20000 #takes nearly 50+h for fully train
    display_step = 5
    test_step = 1000
    total_batch = 50
    
    print("Starting training...")
    learning_rate_current = 0.001
    
    for epoch in range(training_epochs):
        if epoch > 0 and epoch % 2000 == 0:
            learning_rate_current = learning_rate_current / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_current
        
        model.train()
        avg_cost = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= total_batch:
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            avg_cost += loss.item() / total_batch
        
       
        if epoch % display_step == 0:
            print(f"Epoch: {epoch+1:04d}, cost= {avg_cost:.9f}")
            
            # TEST
            test_number = 10000 if epoch % test_step == 0 else 1000
            test_dataset = OFDMDataset(channel_test, SNRdb=SNRdb, dataset_size=test_number)
            test_loader = DataLoader(test_dataset, batch_size=test_number, shuffle=False, num_workers=0)
            
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss = criterion(output, target).item()
                    test_ber = bit_error_rate(output, target).item()
                    
                    print(f"OFDM Detection QAM output number is {16}, SNR = {SNRdb}, "
                          f"Num Pilot = {P}, prediction and the mean error on test set are: "
                          f"{test_loss:.6f}, {test_ber:.6f}")
                    break
            
            # EVAL
            model.eval()
            with torch.no_grad():
                train_data, train_target = next(iter(train_loader))
                train_data, train_target = train_data.to(device), train_target.to(device)
                train_output = model(train_data)
                train_loss = criterion(train_output, train_target).item()
                train_ber = bit_error_rate(train_output, train_target).item()
                
                print(f"prediction and the mean error on train set are: {train_loss:.6f}, {train_ber:.6f}")
    
    print("optimization finished")
    
    # MODEL SAVE
    torch.save(model.state_dict(), 'ofdm_model.pth')
    return model

def evaluate_model():
    """evaluate at the different SNR """
    # loading
    input_size = 256 + len(channel_train[0])
    model = OFDMNet(input_size).to(device)
    model.load_state_dict(torch.load('ofdm_model.pth', map_location=device))
    model.eval()
    
    BER = []
    SNR_range = range(5, 30, 5)
    
    print("Evaluating model across different SNR values...")
    for SNR in SNR_range:
        print(f"Evaluating SNR: {SNR} dB")
        
        # create test dataset
        test_dataset = OFDMDataset(channel_test, SNRdb=SNR, dataset_size=10000)
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
        
        total_ber = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                ber = bit_error_rate(output, target).item()
                total_ber += ber
                break
        
        BER.append(total_ber)
        print(f'SNR {SNR} dB: BER = {total_ber:.6f}')
    
    # save the output result
    BER_array = np.array(BER)
    sio.savemat('BER.mat', {'BER': BER_array})
    print(f"BER results: {BER}")
    print("Results saved to BER.mat")
    
    return BER

if __name__ == "__main__":
    # training
    print("=== Training Phase ===")
    model = train_model()
    
    # evaluate model
    print("\n=== Evaluation Phase ===")
    ber_results = evaluate_model()
