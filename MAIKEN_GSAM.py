import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from connect import init_cognite_connect
from GroundingDINO.groundingdino.util.inference import annotate, predict
from PIL import Image
import io
import GroundingDINO.groundingdino.datasets.transforms as T
from typing import Tuple
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 
import os
import torch 
from PIL import Image, ImageDraw
import numpy as np
from torchvision.ops import box_convert
from cognite.client.data_classes import AnnotationFilter, Annotation


def load_image_pil(image_source: Image.Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = image_source.convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def get_gsam_annotations(file_id, asset_prompt, client, DEVICE, groundingdino_model):
    # Set thresholds
    TEXT_PROMPT = asset_prompt
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    
    # Download and load the image
    temp_png = client.files.download_bytes(id=file_id)
    image = Image.open(io.BytesIO(temp_png))
    image_source, image = load_image_pil(image)
    
    # Predict using the GroundingDINO model
    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_THRESHOLD, 
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )
    
    return boxes, logits, phrases, image_source, image

def annotate_and_save_image(file_id, boxes, logits, phrases, image_source):
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    
    # Convert BGR to RGB
    annotated_frame = annotated_frame[..., ::-1]
    
    # Save the resulting image
    save_folder = "GSAM_PROMPT_RESULT"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{file_id}.jpg")
    
    result_image = Image.fromarray(annotated_frame)
    result_image.save(save_path)
    print(f"Saved annotated image to {save_path}")


def convert_tensor_box_to_pixel(boxes,img_width,img_height):
    scaled_boxes = boxes * torch.tensor([img_width, img_height, img_width, img_height])
    converted_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return converted_boxes

def get_tensor_center(box):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center
def get_cdf_center(annotation,img_width, img_height):
    text_region = annotation.data["textRegion"]
    x_min = text_region["xMin"] * img_width
    x_max = text_region["xMax"] * img_width
    y_min = text_region["yMin"] * img_height
    y_max = text_region["yMax"] * img_height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center
def get_shortest_distance(cdf_annotation, gsam_tensor_boxes, img_width, img_height):
    cdf_center = get_cdf_center(cdf_annotation, img_width, img_height)
    gsam_centers = [get_tensor_center(box) for box in gsam_tensor_boxes]
    distance = np.inf
    for center in gsam_centers:
        current_distance = np.sqrt((cdf_center[0] - center[0]) ** 2 + (cdf_center[1] - center[1]) ** 2)
        if current_distance < distance:
            distance = current_distance
    return distance

def match_cdf_tensor_combo(cdf_annotations, tensor_boxes, tensor_descriptions, relevant_cdf_assets, relevant_image, threshold=150):
    """
    Match CDF annotations with tensor boxes based on descriptions and distances.
    
    Args:
    - cdf_annotations: List of CDF annotations.
    - tensor_boxes: Tensor of bounding boxes.
    - tensor_descriptions: List of descriptions for each tensor.
    - relevant_cdf_assets: List of dictionaries containing relevant CDF assets.
    - relevant_image: Image on which annotations are made.
    - threshold: Distance threshold in pixels for matching (default: 150 pixels).

    Returns:
    - matched_results: List of dictionaries with matched results.
    """
    matched_results = []
    distance_threshold = threshold  # Default 150 pixels - about 20cm
    
    # Get image dimensions
    img_width, img_height = relevant_image.size

    # Convert tensor boxes to pixel values
    converted_boxes = convert_tensor_box_to_pixel(tensor_boxes, img_width, img_height)
    
    # Process each CDF annotation
    for annotation in cdf_annotations.data:
        asset_name = annotation.data["text"].strip().lower()
        
        # Find the relevant asset description
        asset_description = None
        for asset in relevant_cdf_assets:
            if asset.external_id.strip().lower() == asset_name:
                asset_description = asset.description.strip().lower()
                break
        
        if not asset_description:
            print(f"No description found for asset: {asset_name}")
            continue
        
        # Find matching tensor indexes based on description
        matching_indexes = [i for i, desc in enumerate(tensor_descriptions) if desc.strip().lower() == asset_description]
        
        if not matching_indexes:
            print(f"No matching tensor description for asset: {asset_name}")
            continue
        
        # Calculate distances and find the closest match within the threshold
        closest_distance = np.inf
        closest_box = None
        for idx in matching_indexes:
            tensor_box = converted_boxes[idx]
            distance = get_shortest_distance(annotation, [tensor_box], img_width, img_height)
            if distance < closest_distance:
                closest_distance = distance
                closest_box = tensor_box
        
        # Check if the closest distance is within the threshold
        if closest_distance <= distance_threshold:
            matched_results.append({
                "cdf_annotation": annotation,
                "TensorBox": closest_box,
                "TensorDescription": tensor_descriptions[matching_indexes[0]]
            })
        else:
            print(f"No match for: {asset_name}")
    
    return matched_results


def convert_mask_to_vertices(mask, img_width, img_height):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertices = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            # Normalize the coordinates
            x_normalized = x / img_width
            y_normalized = y / img_height
            vertices.append({'x': x_normalized, 'y': y_normalized})
    
    return vertices
    
def get_segment_mask(current_image,current_box, sam_predictor):
    sam_predictor.set_image(current_image)
    masks,_,_ = sam_predictor.predict(box=current_box,multimask_output=False,)
    return masks

def create_assetlink_from_segment(target_asset_id,target_asset_text,text_region,segment_vertices,resource_id):
    vertices = segment_vertices
    link_data = {"assetRef":{"id":target_asset_id}, 
                 "text": target_asset_text, 
                 "textRegion":text_region, 
                 "objectRegion": {
                     "polygon":{
                         "vertices": vertices}
                 }
                } #"externalId":target_asset_ext_id
    assetlink_annotation = Annotation(annotation_type = "images.AssetLink",
                                      data=link_data, 
                                      status = "approved", 
                                      creating_app = "cognite-vision",
                                      creating_app_version = "1.0.0", 
                                      creating_user = "maiken.gjerdseth@cognite.com", 
                                      annotated_resource_type = "file",
                                      annotated_resource_id = resource_id )
    
    return assetlink_annotation
