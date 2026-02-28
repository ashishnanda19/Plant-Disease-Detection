import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ===============================
# CONFIGURATION
# ===============================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple GPU")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

IMG_SIZE = 128
MEAN = [0.5, 0.5, 0.5]
STD = [0.2, 0.2, 0.2]

# ===============================
# CLASS NAMES
# ===============================
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
           'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
           'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
           'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
           'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
           'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
num_classes = len(classes)

# ===============================
# MODEL ARCHITECTURE
# ===============================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "../plant_disease__classification_model.pth"

model = CustomCNN(num_classes=num_classes).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Flexible loading
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('features.0.'): new_key = k.replace('features.0.', 'conv1.')
    elif k.startswith('features.1.'): new_key = k.replace('features.1.', 'bn1.')
    elif k.startswith('features.4.'): new_key = k.replace('features.4.', 'conv2.')
    elif k.startswith('features.5.'): new_key = k.replace('features.5.', 'bn2.')
    elif k.startswith('features.8.'): new_key = k.replace('features.8.', 'conv3.')
    elif k.startswith('features.9.'): new_key = k.replace('features.9.', 'bn3.')
    elif k.startswith('features.12.'): new_key = k.replace('features.12.', 'conv4.')
    elif k.startswith('features.13.'): new_key = k.replace('features.13.', 'bn4.')
    elif k.startswith('classifier.3.'): new_key = k.replace('classifier.3.', 'fc1.')
    elif k.startswith('classifier.6.'): new_key = k.replace('classifier.6.', 'fc2.')
    else: new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()
print("✅ Model Loaded Successfully")

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_image(image_path, top_k=5):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return []

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_idxs = probs.topk(min(top_k, num_classes))
    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    top_classes = [classes[i] for i in top_idxs]
    return list(zip(top_classes, top_probs))

# ===============================
# INFERENCE RUNNER
# ===============================
if __name__ == "__main__":
    TEST_PATH = "dataset/test"

    if os.path.exists(TEST_PATH):
        if os.path.isdir(TEST_PATH):
            test_images = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        else:
            test_images = [TEST_PATH]

        print(f"\nNumber of test images: {len(test_images)}\n")

        for i, img_path in enumerate(test_images):
            predictions = predict_image(img_path, top_k=5)
            if not predictions:
                continue
            
            print(f"Image {i+1} ({os.path.basename(img_path)}):")
            for cls, prob in predictions:
                print(f"  {cls}: {prob:.4f}")
            print("-" * 30)
    else:
        print(f"❌ Target path {TEST_PATH} not found.")
