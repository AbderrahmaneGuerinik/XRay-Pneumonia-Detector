import streamlit as st
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---------------------------
# Transformations pour l'image
# ---------------------------
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionne pour ResNet50
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ---------------------------
# Interface Streamlit
# ---------------------------
st.title("Pneumonia Detection from X-ray Images")

uploaded_image = st.file_uploader(
    'Upload the X-ray image here...', type=["jpg", "jpeg", "png"])

# ---------------------------
# Charger le modèle
# ---------------------------


@st.cache_resource  # évite de recharger le modèle à chaque interaction
def load_model():
    # Utiliser les poids None pour ne pas télécharger de modèle pré-entraîné
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes : Normal / Pneumonia
    model.load_state_dict(torch.load("resnet50_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()

# ---------------------------
# Préparer l'image et faire la prédiction
# ---------------------------
if uploaded_image is not None:
    try:
        image_data = uploaded_image.getvalue()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  # assure RGB
        image_transformed = val_test_transform(
            image).unsqueeze(0)  # batch dimension

        with torch.no_grad():
            output = model(image_transformed)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0, pred].item()
            result = "Pneumonia" if pred == 1 else "Normal"

        st.markdown(
            f"""
            <div style='font-size:22px; font-weight:bold; color:#1f77b4;'>
                Prediction: <span style='color:green;'>{result}</span><br>
                Confidence: <span style='color:orange;'>{confidence*100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error processing image: {e}")
