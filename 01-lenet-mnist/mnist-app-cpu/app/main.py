import base64
import io
import os
import time
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

SHARED_DIR = "/shared_data"
MODEL_PATH = os.path.join(SHARED_DIR, "lenet_mnist.pth")

class LeNet(nn.Module):
    def __init__(self, num_channels=1, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        return self.logsoftmax(self.fc2(x))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

device = torch.device("cpu")
model = LeNet().to(device)

transform = Compose([
    Grayscale(num_output_channels=1),
    Resize((28, 28)),
    ToTensor(),
])

@app.on_event("startup")
def load_trained_model():
    print(f"Waiting for trained model file at {MODEL_PATH}...")
    while not os.path.exists(MODEL_PATH):
        time.sleep(2)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded into memory successfully!")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
    request=request, 
    name="index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400,
                            detail="That file couldn't be read as an image. Try a PNG or JPG.")
    image = ImageOps.invert(image)  # Invert if white background with black text

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.exp(model(tensor))[0]
        prediction = probs.argmax().item()
        confidence = probs[prediction].item()

    # The 28x28 image the network actually received, for display in the page.
    buf = io.BytesIO()
    to_pil_image(tensor[0].cpu()).save(buf, format="PNG")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": [round(p, 4) for p in probs.tolist()],
        "model_input_png": base64.b64encode(buf.getvalue()).decode(),
    }