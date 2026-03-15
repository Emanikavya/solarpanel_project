import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# ResNet Custom Model
# -----------------------------------
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

        # EXACT SAME CLASSIFIER USED DURING TRAINING
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

feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
import numpy as np
import cv2
import os

X = []
y = []

for image_path in dataset_images:

    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))
    img = img/255.0

    img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(img)

    features = features.view(-1).numpy()

    X.append(features)

    y.append([power_loss, efficiency, lifespan])
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)

model = MultiOutputRegressor(xgb)

model.fit(X, y)
import joblib

joblib.dump(model,"solar_xgb_model.pkl")