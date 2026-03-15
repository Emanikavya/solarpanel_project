from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import joblib
import requests
from io import BytesIO

app = Flask(__name__)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# ResNet Custom Model
# -------------------------------
class ResNetCustom(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        base_model = models.resnet50(weights="IMAGENET1K_V1")

        for param in base_model.parameters():
            param.requires_grad = False

        for name, param in base_model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True

        self.features = nn.Sequential(*list(base_model.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(

            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512,num_classes)
        )

    def forward(self,x):

        x = self.features(x)

        pooled = self.pool(x)

        feature_vector = torch.flatten(pooled,1)

        out = self.classifier(feature_vector)

        return out, feature_vector


# -------------------------------
# Load Models
# -------------------------------
resnet_model = ResNetCustom(6).to(device)
resnet_model.load_state_dict(torch.load("solar_model.pth", map_location=device))
resnet_model.eval()

xgb_model = joblib.load("solar_regression_model.pkl")

print("Models loaded successfully")


# -------------------------------
# Class Labels
# -------------------------------
class_names = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]


# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# Prediction
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----- Load Image -----
        if "image" in request.files:

            file = request.files["image"]
            img = Image.open(file).convert("RGB")

        elif "image_url" in request.form:

            url = request.form["image_url"]
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")

        else:
            return jsonify({"error":"No image provided"})


        # ----- Preprocess -----
        img_tensor = transform(img).unsqueeze(0).to(device)


        # ----- ResNet Prediction -----
        with torch.no_grad():

            output, features = resnet_model(img_tensor)

            probs = F.softmax(output, dim=1)

            confidence, pred = torch.max(probs,1)


        defect = class_names[pred.item()]
        confidence = confidence.item()


        # reject unrelated images
        if confidence < 0.40:
            return jsonify({"error":"Not a solar panel image"})


        # ----- Prepare Features for XGBoost -----
        features = features.cpu().numpy().reshape(1,-1)


        # ----- Regression Prediction -----
        prediction = xgb_model.predict(features)

        power_loss = float(prediction[0][0])
        efficiency = float(prediction[0][1])
        lifespan = float(prediction[0][2])


        status = "Healthy" if defect=="Clean" else "Defective"


        return jsonify({

            "status":status,
            "defect":defect,
            "confidence":round(confidence*100,2),

            "power_loss":round(power_loss,2),
            "efficiency":round(efficiency,2),
            "lifespan":round(lifespan,1)

        })


    except Exception as e:

        return jsonify({
            "error":str(e)
        })


# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)