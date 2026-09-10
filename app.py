from torch import nn
import torch
from pathlib import Path
import os
from typing import Tuple
from model import create_model
from timeit import default_timer as timer
import streamlit as st
from PIL import Image


model_path = Path(__file__).parent / "effb2_food101_final.pth"

checkpoint = torch.load(model_path, map_location="cpu")

classes = checkpoint["classnames"]

num_classes = checkpoint["num_classes"]

model_state_dict = checkpoint["model_state_dict"]

model, train_transforms, test_transforms = create_model(num_classes)

model.load_state_dict(model_state_dict)

def predict(img) -> Tuple[str, float, float]:    
    
    time_start = timer()
    
    img_transformed = test_transforms(img)
    
    img_transformed = img_transformed.unsqueeze(dim=0)
    
    model.eval()
    with torch.inference_mode():
        y_logits = model(img_transformed)
        
        y_probs = torch.softmax(y_logits, dim=1)
        
        y_pred = torch.argmax(y_probs, dim=1)

        confidence = y_probs.max(dim=1).values
        
        top_probs, top_indices = torch.topk(y_probs, k=5, dim=1)

    pred_class = classes[y_pred.item()]
    confidence = confidence.item()
    
    top_probs = top_probs.squeeze(0)
    top_indices = top_indices.squeeze(0)
    
    top5 = []
    for prob, index in zip(top_probs, top_indices):
        class_name = classes[index]
        probability = prob.item()
        top5.append((class_name, probability))
        
    time_end = timer()
    time_elapsed = time_end - time_start
    
    return pred_class, confidence, time_elapsed, top5

# print(predict(Image.open("examples/863850.jpg")))
def main():
    st.set_page_config("foodVision",
                    page_icon="🍔", 
                    layout="wide")
    st.title("🍔FoodVision")
    st.write("Food vision classifier powered by EfficientNet-B2")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Upload an image")
        uploaded_file = st.file_uploader(label="choose an image", 
                                        type=["jpg","png","jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image")
            st.write("Image uploaded successfully! 🎉")    
            pred_class, conf, inference_time, top5 = predict(image)
    with col2:
        if uploaded_file is not None:
            st.subheader("Prediction")
            st.write(f"**Class:** {pred_class.replace("_"," ").capitalize()}")
            st.write(f"**Confidence:** {conf:.2%}")
            st.write(f"**time prediction:** {inference_time:.4f} s.")

            st.subheader("5 Top Predictions")
            for classname, probability in top5:
                st.write(f"**{classname.replace("_"," ").capitalize()}**",
                        f"- {probability:.2%}")
                st.progress(probability)
    st.markdown(
        """
        <style>
        .footer {
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #262730;
            z-index: 999;
        }

        .footer p {
            margin: 2px;
            font-size: 13px;
            color:#FAFAFA99
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
                <footer class="footer">
                    <p>Built with PyTorch & Streamlit</p>
                    <p>ّFoodVision • EfficientNet-B2 • Food101</p>
                    <a href="https://github.com/MohammadNazari98">Developed by MohammadDev<a/>
                </footer>
                """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
