from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mri_loaders(train_path, test_path, batch_size=32):
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader, train_data.classes


class EEGDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)

            for file in os.listdir(cls_path):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(cls_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y)

        return x, y


def get_eeg_loader(path, batch_size=32):
    dataset = EEGDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, dataset.classes