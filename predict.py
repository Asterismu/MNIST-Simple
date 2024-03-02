import torch
from model import Net
from PIL import Image
import torchvision.transforms as transforms

model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
def preprocess(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)

def predict(image_path):
    image_tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

if __name__ == '__main__':
    image_path = 'C:\\Users\\Asterism\\Pictures\\sample.png'
    
    prediction = predict(image_path)
    print(f'resultDigit: {prediction}')