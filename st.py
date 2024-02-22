import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests

# Load DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Streamlit app
def main():
    st.title("Object Detection App")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Detect objects on button click
        if st.button("Detect Objects"):
            # Convert uploaded image to PIL Image
            image = Image.open(uploaded_image)

            # Process image with DETR model
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Convert outputs to COCO API format
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 20), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}",
                          fill="red")

            # Display image with bounding boxes
            st.image(image, caption='Image with Bounding Boxes', use_column_width=True)

if __name__ == "__main__":
    main()
