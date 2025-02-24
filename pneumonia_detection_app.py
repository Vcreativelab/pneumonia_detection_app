"""
Created on Wed Dec 14 18:45 2022
@author: VirtualCreativeLab
Required Packages: requirements.txt
CNN Model: model.py
"""

# Core Pkgs
import streamlit as st
st.set_page_config(page_title="Pneumonia Detection Tool", page_icon="pneumonia.png", layout='centered',
                   initial_sidebar_state='auto')

# import os
import time


# Viz Pkgs
import cv2
import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np


# AI Pkgs
import torch
import torchvision.transforms.functional as fn
from model import PneumoniaClassifier
from xplainer import Xplainer, cam


def main():
    """Simple Tool for Pneumonia Detection from Chest X-Ray"""
    html_templ = """
    <div style="background-color:#E3F2FD;padding:15px;border-radius:10px;">
        <h1 style="color:#0D47A1;text-align:center;">Pneumonia Detection Tool</h1>
    </div>
    """


    st.markdown(html_templ, unsafe_allow_html=True)
    st.write("A simple proposal for Pneumonia Diagnosis powered by Deep Learning and Streamlit")


    st.sidebar.image("pneumonia.png", width=300)
    image_file = st.sidebar.file_uploader("Upload an X-Ray Image (DICOM)", type=['dcm'])
    if image_file is not None:
        # our_image = Image.open(image_file)
        ds = dicom.dcmread(image_file)
        image = ds.pixel_array
        # convert to PIL image
        our_image = Image.fromarray(image)

        if st.sidebar.button("Image Preview"):
            st.sidebar.image(our_image, width=300)

        activities = ["Image Enhancement", "Diagnosis", "Disclaimer and Info"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        if choice == 'Image Enhancement':
            st.subheader("Image Enhancement")

            enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Contrast", "Brightness"])

            if enhance_type == 'Contrast':
                c_rate = st.slider("Contrast", 0.5, 5.0)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, use_column_width=True)

            elif enhance_type == 'Brightness':
                c_rate = st.slider("Brightness", 0.5, 5.0)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output, width=600, use_column_width=True)

            else:
                st.text("Original Image")
                st.image(our_image, width=600, use_column_width=True)

        elif choice == 'Diagnosis':
            if st.sidebar.button("Diagnosis"):
                st.text("Chest X-Ray")
                # Image to Black and White
                new_img = np.array(our_image.convert('RGB')) #our image is binary we have to convert it in array
                new_img = cv2.cvtColor(new_img,1) # 0 is original, 1 is grayscale
                grey = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
                st.image(grey, use_column_width=True)
                # Image preprocessing
                img = cv2.resize(grey,(224,224)).astype(np.float16)
                img = img/255.
                tensor_image = fn.to_tensor(img)
                mean = tensor_image.mean()
                std = tensor_image.std()
                img = fn.normalize(tensor_image, mean=mean, std=std)
                # Pre-Trained Model Importing

                # Load pre-trained classification model
                checkpoint_path = 'checkpoint/weights_1.ckpt'
                model = PneumoniaClassifier.load_from_checkpoint(checkpoint_path, strict=False)
                model.eval()  # switch to eval model
                # Use strict to prevent pytorch from loading weights for self.feature_map
                # Load pre-trained model for interpretability
                explainer = Xplainer.load_from_checkpoint(checkpoint_path, strict=False)
                explainer.eval()
                # Disable gradient and get prediction
                with torch.no_grad():  # deactivate grad
                   img = img.float().unsqueeze(0)
                   pred = torch.sigmoid(model(img)[0].cpu()).numpy()

                # Display progress bar with text
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()

                # Diagnosis with probabilities
                diagnosis_proba = pred.item()
                probability_pne = diagnosis_proba*100
                probability_no_pne = (1-diagnosis_proba)*100
                diagnosis = int(diagnosis_proba>0.5)
                # Diagnosis Cases: No-Pneumonia=0, Pneumonia=1
                if diagnosis == 0:
                   st.sidebar.success("DIAGNOSIS: NO PNEUMONIA (Probability: %.2f%%)" % (probability_no_pne))
                else:
                   st.sidebar.error("DIAGNOSIS: PNEUMONIA (Probability: %.2f%%)" % (probability_pne))

                activation_map, preds = cam(explainer, img)  # Compute the Class activation map given the subject
                img = img[0]  # remove channel dimension
                # Resize the activation map of size 7x7 to the original image size (224x224)
                # unsqueeze(0) because .resize expects a channel dimension
                heatmap = fn.resize(activation_map.unsqueeze(0), (img[0].shape[0], img[0].shape[1]))[0]  # remove channel dim,again

                # Create a figure
                fig, axis = plt.subplots(1, 2, figsize=(10, 7))

                axis[0].imshow(img[0], cmap="bone")
                # Overlay the original image with the upscaled class activation map
                axis[1].imshow(img[0], cmap="bone")
                axis[1].imshow(heatmap, alpha=0.5, cmap="jet")

                st.subheader('Interpretability', divider='rainbow')
                outcome = (pred > 0.5).item()
                if str(outcome) == "True":
                    color = "red"
                else:
                    color = "green"
                st.subheader(f"Pneumonia :{color}[{outcome}]")
                st.pyplot(fig)
                time.sleep(0.05)

                st.warning("This Web App is just a DEMO application of Deep Learning to Medical Imaging. It is not clinically validated!")
        # Disclaimer & Info
        else:
            st.subheader("Disclaimer and Info")
            st.subheader("Disclaimer")
            st.write(
                "**This Tool is just a DEMO about Artificial Neural Networks & its diagnosis has no clinical value**")

            st.write("This Tool gets inspiration from the following works:")
            st.write(
                "- [Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)")
            st.write(
                "- [Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)")
            st.write(
                "- [Class Activation Maps (CAM)](https://arxiv.org/abs/1512.04150)")
            st.write(
                "For the training, we used the RSNA (Radiological Society of North America) Pneumonia Dataset, which comprises of 26684 X-Ray Images. "
                "Of these, 20672 images show signs of pneumonia."
                " The dataset was split into 24k train images and 2684 val images. We also applied some data augmentation techniques (random rotation, traslation, scale & resized crop).")
            st.write(
                "Validation accuracy was 76.5%, recall 85.4% and precision only 48.8%. In fact, we got 542 'False Positive', patients without pneumonia classified as with pneumonia and 88 cases of 'False Negative', patients classified as healthy that actually are infected by pneumonia. All is all, is better to have a high false positive rate & a low false negative rate.")
    # Author's Note
    if st.sidebar.button("About the Author"):
        st.sidebar.subheader("_Pneumonia Detection Tool_")
        st.sidebar.markdown("by [VirtualCreativeLab](https://github.com/Vcreativelab?tab=repositories)")
        st.sidebar.markdown("[Drop an email to the author](mailto:labvcreative@gmail.com)")
        st.sidebar.text("All Rights Reserved (2023)")


if __name__ == '__main__':
    main()
