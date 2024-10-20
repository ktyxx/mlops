import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import urllib.request
from urllib.error import HTTPError
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# dataset 저장경로 및 체크포인트 경로
dataset_dir = '/mnt/data'
CHECKPOINT_PATH = './checkpoints'

# pretrained 모델 다운로드
def download_pretrained_models():
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
    pretrained_files = ["ResNet.ckpt"]
    
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
                print(f"Downloaded {file_name} successfully.")
            except HTTPError as e:
                print(f"Error downloading {file_name}: {e}")

# pretrained 모델 로드
def load_pretrained_model():
    print("Loading pretrained ResNet model...")
    # pretrained=True
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)  
    
    return model

# MNIST dataset 로드 및 데이터 로더 설정
def get_data_loaders():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 1채널->3채널(resnet18 모델은 기본 입력값이 3채널)
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    
    print("Loading MNIST dataset...")
    
    # 학습 dataset
    train_dataset = datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 테스트 dataset
    test_dataset = datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# PyTorch Lightning 학습 클래스(ResNet 모델 사용)
class MNISTResNet(pl.LightningModule):
    def __init__(self):
        super(MNISTResNet, self).__init__()
        self.model = load_pretrained_model()  
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        print(f"Epoch [{self.current_epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item()}")
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    # 학습 종료후 모델 저장
    def on_train_end(self):
        model_save_path = '/mnt/data/finetuned_resnet18.pth' 
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")




if __name__ == "__main__":    
    download_pretrained_models()
    
    train_loader, test_loader = get_data_loaders()
    
    model = MNISTResNet()
    
    # GPU가 없는 CPU 환경에서 학습, 빠른 테스트 진행 위해 epoch 1로 지정
    trainer = Trainer(max_epochs=1, accelerator="cpu")
    
    trainer.fit(model, train_loader)
