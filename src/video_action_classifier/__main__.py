import argparse
import os

from decord import cpu
from decord import VideoReader
import numpy as np
import torch
from transformers import XCLIPModel
from transformers import XCLIPProcessor


def analyze_video_in_chunks(video_path,
                            custom_labels,
                            chunk_duration_sec=2.0,
                            clip_len=8):
    # 1. Load the Zero-Shot X-CLIP model ONCE for efficiency
    print("Loading model...")
    model_name = "microsoft/xclip-base-patch32"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)

    # 2. Load video and calculate chunk properties
    print(f"Loading video: {video_path}")
    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    fps = videoreader.get_avg_fps()
    total_frames = len(videoreader)

    # Calculate how many frames are in one chunk
    frames_per_chunk = int(fps * chunk_duration_sec)

    print(f"Video FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Analyzing in {chunk_duration_sec}s chunks...\n")

    results = []

    # 3. Iterate through the video in chunks
    for start_frame in range(0, total_frames, frames_per_chunk):
        end_frame = min(start_frame + frames_per_chunk, total_frames)

        # Skip the last chunk if it's too small to extract the required clip_len (8 frames)
        if end_frame - start_frame < clip_len:
            continue

        # Sample 'clip_len' (8) frames evenly across this specific chunk
        indices = np.linspace(start_frame, end_frame - 1,
                              num=clip_len).astype(np.int64)
        frames = videoreader.get_batch(indices).asnumpy()

        # Preprocess both the chunk's frames AND your custom text labels
        inputs = processor(text=custom_labels,
                           videos=list(frames),
                           return_tensors="pt",
                           padding=True)

        # 4. Run inference for this chunk
        with torch.no_grad():
            outputs = model(**inputs)

        # 5. Calculate probabilities
        probs = outputs.logits_per_video.softmax(dim=1)

        # Get the highest probability prediction
        predicted_class_idx = probs.argmax(-1).item()
        predicted_class = custom_labels[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()

        # Calculate timestamps for output
        start_time = start_frame / fps
        end_time = end_frame / fps

        chunk_result = {
            "start": start_time,
            "end": end_time,
            "action": predicted_class,
            "confidence": confidence
        }
        results.append(chunk_result)

        print(
            f"[{start_time:05.2f}s - {end_time:05.2f}s] Action: {predicted_class} (Confidence: {confidence:.2%})"
        )

    return results


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze video chunks for specific actions using X-CLIP.")
    parser.add_argument("video", help="Path to the video file to analyze")
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.txt",
        help="Path to a text file containing one action label per line (default: labels.txt)"
    )
    parser.add_argument("--chunk_duration",
                        type=float,
                        default=2.0,
                        help="Duration of each chunk in seconds (default: 2.0)")

    args = parser.parse_args()

    # Ensure the labels file exists
    if not os.path.exists(args.labels):
        print(f"Error: The labels file '{args.labels}' was not found.")
        exit(1)

    # Read the labels from the text file
    with open(args.labels, "r") as file:
        # Read lines, remove extra whitespace/newlines, and ignore blank lines
        my_actions = [line.strip() for line in file if line.strip()]

    if not my_actions:
        print(
            f"Error: The file '{args.labels}' is empty. Please add at least one label."
        )
        exit(1)

    print(f"Loaded {len(my_actions)} labels from '{args.labels}'")

    # Run the analysis using the arguments provided
    timeline = analyze_video_in_chunks(args.video,
                                       my_actions,
                                       chunk_duration_sec=args.chunk_duration)


# Run the script
if __name__ == "__main__":
    main()
