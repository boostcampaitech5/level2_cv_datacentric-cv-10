import time

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from detect import detect
from model import EAST

st.set_page_config(initial_sidebar_state="collapsed")
st.title("OCR Prediction Result")


def load_model(model_file):
    model = EAST(pretrained=False).to("cuda")
    model.load_state_dict(torch.load(model_file, map_location="cuda"))
    model.eval()

    return model


def do_inference(model, img, input_size=2048):
    image_fnames, by_sample_bboxes = [], []

    # img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0)

    image_fnames, by_sample_bboxes = ["test"], []
    images = []
    images.append(img)

    by_sample_bboxes.extend(detect(model, images, input_size))
    # st.text(by_sample_bboxes)
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {
            idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)
        }
        ufo_result["images"][image_fname] = dict(words=words_info)

    return ufo_result


def main():
    st.error("Best EAST Model Importation")
    uploaded_model = st.file_uploader(
        "Best Model", accept_multiple_files=False, type=["pth", "pt"]
    )
    if uploaded_model is not None:
        model = load_model(uploaded_model)
        model.eval()

    st.warning("Choice your own Sample Test Image")
    uploaded_image = st.file_uploader("Chosen Image", accept_multiple_files=False)
    if uploaded_image is not None and uploaded_model is not None:
        img = np.array(Image.open(uploaded_image))
        fin_img = img.copy()

        st.success("Now Inference In Progress")
        ufo_result = do_inference(model, img)
        height, width, channel = img.shape
        st.info(f"Sample Test Image Size = [{channel}, {height}, {width}]")

        for _, v in ufo_result["images"]["test"]["words"].items():
            v = v["points"]
            v.append(v[0])
            cv2.polylines(fin_img, [np.array(v, dtype=np.int32)], True, (0, 0, 255), 2)
        st.image(fin_img)


if __name__ == "__main__":
    main()
