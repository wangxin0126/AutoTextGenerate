import datetime
import torch

def get_time():
    time_now = datetime.datetime.now()
    return time_now.strftime("%b %d %Y %H:%M:%S")

def check_gpu():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        device = torch.device("cpu")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device