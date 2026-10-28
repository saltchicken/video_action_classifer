import torch
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel
from decord import VideoReader, cpu

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    # Calculates the exact frame indices to extract
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def classify_custom_action(video_path, custom_labels):
    # 1. Load the Zero-Shot X-CLIP model
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    # 2. Extract frames
    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    video_length = len(videoreader)
    
    # X-CLIP uses 8 frames to understand temporal movement
    clip_len = 8 
    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=4, seg_len=video_length)
    frames = videoreader.get_batch(indices).asnumpy()

    # 3. Preprocess both the video frames AND your custom text labels
    inputs = processor(
        text=custom_labels, 
        videos=list(frames), 
        return_tensors="pt", 
        padding=True
    )

    # 4. Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 5. Calculate probabilities across your custom labels
    # logits_per_video represents the similarity score between the video and each text prompt
    probs = outputs.logits_per_video.softmax(dim=1)
    
    # Get the highest probability prediction
    predicted_class_idx = probs.argmax(-1).item()
    predicted_class = custom_labels[predicted_class_idx]
    confidence = probs[0][predicted_class_idx].item()
    
    print(f"Video: {video_path}")
    print(f"Predicted Action: {predicted_class} (Confidence: {confidence:.2%})")
    
    return predicted_class

# Run the script
if __name__ == "__main__":
    video_file = "test3.mp4"
    
    # Define EXACTLY what you are looking for in plain English
    my_actions = [
        "a person sticking their tongue out",
        "a person puffing their cheeks",
        "a person raising their eyebrows",
        "a person crossing their eyes",
        "a person with a neutral resting face"
    ]
    
    classify_custom_action(video_file, my_actions)
