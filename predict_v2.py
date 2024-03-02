from tkinter import font
import torch
from model import Net
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog

# Initialize the model
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Preprocessing function
def preprocess(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image_path):
    image_tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

# create the UI interface
class App:
    def __init__(self, root):
        label_font = font.Font(family="Helvetica", size=18)
        button_font = font.Font(family="Helvetica", size=16)
        self.root = root
        self.root.title("NN Network MNIST(SINGLE) Prediction")
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        # self.canvas.font = label_font
        self.label = tk.Label(root, text="Click 'Choose Image' to predict", width=50, height=4, font=label_font)
        self.label.pack()
        self.button = tk.Button(root, text="Choose Image", command=self.choose_image, font=button_font)
        self.button.pack()
        


    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            prediction = predict(file_path)
            self.show_image(file_path)
            self.label.config(text=f'Predicted digit: {prediction}')

    def show_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(200, 200, image=photo)
        self.canvas.image = photo

# run the UI interface
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()