import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Net

def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    # Load the trained model
    model = Net()
    model.load_state_dict(torch.load(r'mnist_model.pth', map_location=torch.device('cpu')))

    # Define the MNIST dataset transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to the same size as SVHN images
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize using MNIST's mean and std
    ])

    # Load the MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    accuracy = test_model(model, test_loader)
    print(f'Model accuracy on MNIST test set: {accuracy * 100:.2f}%')