import gradio as gr
import numpy as np
import cv2
import torch
from typing import List
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from model import *

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Checkpoints path
checkpoints_path = "checkpoints/checkpoint_epoch_9.pt"

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
# model = OwlViTForObjectDetectionModel.from_pretrained("google/owlvit-base-patch32")
# model.load_state_dict(torch.load(checkpoints_path)['model_state_dict'])
model.to(device)
model.eval()



def text_guided_inference(img, text_queries, score_threshold):
    text_queries = text_queries
    text_queries = text_queries.split(",")

    #target_sizes = torch.Tensor([img.size[::-1]])
    target_sizes = torch.Tensor([img.shape[:2]])
    inputs = processor(text=text_queries, images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu() 
    results = processor.post_process_object_detection(outputs=outputs, threshold = score_threshold,target_sizes=target_sizes)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]

        if score >= score_threshold:
            img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 1)
            if box[3] + 25 > 768:
                y = box[3] - 10
            else:
                y = box[3] + 25
                
            img = cv2.putText(
                img, text_queries[label], (box[0], y), font, 0.7, (255,0,1), 1, cv2.LINE_AA
            )
    return img


def image_guided_inference(img, query_img, score_threshold, nms_threshold):
    target_sizes = torch.Tensor([img.shape[:2]])
    inputs = processor(query_images= query_img, images = img, return_tensors= "pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

    results = processor.post_process_image_guided_detection(outputs = outputs,
                                                            threshold = score_threshold,
                                                            nms_threshold = nms_threshold,
                                                            target_sizes = target_sizes)
    

    boxes, scores = results[0]["boxes"], results[0]["scores"]
    img = np.asarray(img)


    for box, score in zip(boxes, scores):
        box = [int(i) for i in box.tolist()]

        if score >= score_threshold:
            img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 1)
            if box[3] + 25 > 768:
                y = box[3] - 10
            else:
                y = box[3] + 25
    
    return img

description = """
Gradio demo for text-guided and image-guided object detection with OWL-ViT - 
<a href="https://huggingface.co/docs/transformers/main/en/model_doc/owlvit">OWL-ViT</a>, 
introduced in <a href="https://arxiv.org/abs/2205.06230">Simple Open-Vocabulary Object Detection
with Vision Transformers</a>. 
\n\nYou can use OWL-ViT to query images with text descriptions of any object or alternatively with an 
example / query image of the target object. To use it, simply upload an image and a query image that only contains the object
 you're looking for. You can also use the score and non-maximum suppression threshold sliders to set a threshold to filter out 
 low probability and overlapping bounding box predictions.
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # DEMO FOR TEXT-GUIDED AND IMAGE-GUIDED OBJECT DETECTION WITH OWL-VIT MODEL.
        This model was introduced in <a href="https://arxiv.org/abs/2205.06230">Simple Open-Vocabulary Object Detection with Vision Transformers</a>. 
        
        This project demo showcases a finetuned OwL-ViT model for content moderation, trained on a bad content dataset (cigarette, drug, alcohol, gun). The model can perform both text-guided and image-guided object detection, enabling users to identify and remove inappropriate content from images and videos.
        
        For example, users can input a text query such as "cigarette" or "gun" and the model will highlight all occurrences of those objects in the image. Alternatively, users can provide the model with an image and it will identify the similar content present. Moreover, users can also use the score and non-maximum suppression threshold sliders to set a threshold to filter out 
        low probability and overlapping bounding box predictions.
        
        This demo has the potential to elevate content moderation by providing a fast and efficient way to identify and remove inappropriate content. It can be used by social media platforms, online retailers, and other organizations to protect their users from inappropriate content. 
        """)
    with gr.Tab("text-guided object detection"):
        with gr.Column():
            img = gr.Image(label = "Upload your target image here", show_label= True)
            text_queries = gr.Textbox(label = "Please enter your comma-separated text prompts here")
            txt_score_threshold = gr.Slider(minimum=0, maximum=1, value=0.2, label="Adjust confidence threshold as needed")

        with gr.Column():
            
            inference_text_btn = gr.Button("Run detection", variant = "primary")
            output_img = gr.Image(label = "Output of detector is showed below", interactive = False)
        

    with gr.Tab("image-guided object detection"):
        
        with gr.Column():
            input_img = gr.Image(label = "Upload your target image here", show_label= True)
            query_img = gr.Image(label = "Upload your query image here", show_label= True)
            img_score_threshold = gr.Slider(minimum=0, maximum=1, value=0.2, label="Adjust confidence score's threshold as needed")
            nms_threshold = gr.Slider(minimum=0, maximum=1, value=0.3, label="Adjust non-maximum suppression threshold as needed")

        with gr.Column():
            
            inference_img_btn = gr.Button("Run detection", variant= "primary")
            out_img = gr.Image(label = "Output of detector is showed below", interactive = False)

    inference_text_btn.click(text_guided_inference, inputs= [img, text_queries, txt_score_threshold], outputs= [output_img])
    inference_img_btn.click(image_guided_inference, inputs= [input_img, query_img, img_score_threshold, nms_threshold], outputs= [out_img])

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(share= True)
