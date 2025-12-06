import pandas as pd
import argparse
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

### Configuration

def load_argparser():
    parser = argparse.ArgumentParser(description="AML 2025: Feathers in focus")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-model", action="store_true")
    return parser

### Image loading

TRANSFORM_DEFAULT = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((128, 128)),
        v2.ToDtype(tc.float32, scale=True),
    ]
)

class ImageClassification(Dataset):
    
    def __init__(
        self,
        csv_file: Path,
        transform: v2.Transform = TRANSFORM_DEFAULT,
    ):
        self.df = pd.read_csv(csv_file, header=0)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        label = int(row.label)
        path = row.image_path
        
        full_path = Path("./data/") / path.lstrip("/")
        
        image = Image.open(full_path).convert("RGB")
        image = self.transform(image)
        return idx, image, label, path
        
        

### Model generation

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1) # 3 input channels, as input is RGB
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(246016, 128) # 64 * 62 * 62 = 246016
        self.fc2 = nn.Linear(128, 201) # 200 output classes
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = tc.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
### Training
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (_, data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                .format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item()))
            if args.dry_run:
                break

### Testing

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with tc.no_grad():
        for _, data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
        .format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)))


### Full model


if __name__ == "__main__":
    parser = load_argparser()
    args = parser.parse_args()
    
    tc.manual_seed(args.seed)
    
    device = tc.device("cpu")
    
    IMAGE_SIZE = 128
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN), (NORM_STD))
        ]
    )
    
    dataset1 = ImageClassification(
        csv_file = "./data/train_images.csv",
        transform = transform
    )
    dataset2 = ImageClassification(
        csv_file = "./data/test_images_path.csv",
        transform = transform
    )
    
    train_loader = tc.utils.data.DataLoader(
        dataset1, 
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = tc.utils.data.DataLoader(
        dataset2, 
        batch_size=args.test_batch_size
    )
    
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        
    if args.save_model:
        tc.save(model.state_dict(), "feathersinfocus_cnn.pt")