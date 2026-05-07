import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Pneumonia Detection System")
st.write("Upload a chest X-ray image to detect pneumonia")

model_path='best_model.pth'

@st.cache_resource
def load_model():
    model=models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1536, 1))

    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
model= load_model()
def predict(image):
    img = image.convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor).squeeze(1)
        prob=torch.sigmoid(output).item()
        return prob
    
image_input=st.file_uploader("Upload chest X-Ray",type=['jpg', 'jpeg', 'png'])
 
if image_input is not None:
    image=Image.open(image_input)
    col1,col2=st.columns(2)

    with col1:
        st.subheader("X-Ray Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        prob = predict(image)

        thrasehold=0.97
        if prob>=thrasehold:
            st.error("PNEUMONIA DETECTED")
        else:
            st.success("Normal")

        st.metric("Confidence", f"{prob:.4f}")
        st.progress(float(prob))


