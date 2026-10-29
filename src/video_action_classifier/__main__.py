import argparse
import os
import sys

from decord import cpu
from decord import VideoReader
import numpy as np
import torch

# ==========================================
# 1. MODEL ABSTRACTIONS (Extracted Logic)
# ==========================================


class BaseActionModel:
    """Base class for any video/image action classification model."""

    def load(self):
        """Loads the model and processor into memory."""
        raise NotImplementedError

    def predict(self, frames, labels):
        """
        Takes a numpy array of frames and a list of string labels.
        Returns a tuple: (predicted_label_string, confidence_float)
        """
        raise NotImplementedError


class XCLIPModelStrategy(BaseActionModel):
    """Implementation for Zero-Shot X-CLIP."""

    def __init__(self, model_name="microsoft/xclip-base-patch32"):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load(self):
        from transformers import XCLIPModel
        from transformers import XCLIPProcessor
        print(f"Loading X-CLIP model ({self.model_name})...")
        self.processor = XCLIPProcessor.from_pretrained(self.model_name)
        self.model = XCLIPModel.from_pretrained(self.model_name)

    def predict(self, frames, labels):
        # Preprocess
        inputs = self.processor(text=labels,
                                videos=list(frames),
                                return_tensors="pt",
                                padding=True)
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate probabilities
        probs = outputs.logits_per_video.softmax(dim=1)

        # Get highest probability
        predicted_class_idx = probs.argmax(-1).item()
        predicted_class = labels[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()

        return predicted_class, confidence


# ==========================================
# 2. CORE LOGIC (Now Model-Agnostic)
# ==========================================


def format_timestamp(seconds):
    """Converts seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_video_in_chunks(video_path,
                            custom_labels,
                            model_strategy: BaseActionModel,
                            chunk_duration_sec=2.0,
                            clip_len=8,
                            threshold=0.0):

    # 1. Load video and calculate chunk properties
    print(f"Loading video: {video_path}")
    videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    fps = videoreader.get_avg_fps()
    total_frames = len(videoreader)

    # Calculate how many frames are in one chunk
    frames_per_chunk = int(fps * chunk_duration_sec)

    print(f"Video FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Analyzing in {chunk_duration_sec}s chunks...\n")

    grouped_results = []
    current_group = None

    # 2. Iterate through the video in chunks
    try:
        for start_frame in range(0, total_frames, frames_per_chunk):
            end_frame = min(start_frame + frames_per_chunk, total_frames)

            # Skip if chunk is too small
            if end_frame - start_frame < clip_len:
                continue

            # Sample frames evenly
            indices = np.linspace(start_frame, end_frame - 1,
                                  num=clip_len).astype(np.int64)
            frames = videoreader.get_batch(indices).asnumpy()

            # 3. USE THE ABSTRACTED MODEL STRATEGY
            predicted_class, confidence = model_strategy.predict(
                frames, custom_labels)

            # 4. Group dynamically
            start_time = start_frame / fps
            end_time = end_frame / fps

            if confidence >= threshold:
                if current_group is None:
                    current_group = {
                        "start": start_time,
                        "end": end_time,
                        "action": predicted_class,
                        "confidences": [confidence]
                    }
                else:
                    if predicted_class == current_group["action"] and abs(
                            start_time - current_group["end"]) < 0.1:
                        current_group["end"] = end_time
                        current_group["confidences"].append(confidence)
                    else:
                        avg_conf = sum(current_group["confidences"]) / len(
                            current_group["confidences"])
                        start_str = format_timestamp(current_group["start"])
                        end_str = format_timestamp(current_group["end"])

                        out_str = f"[{start_str} - {end_str}] Action: {current_group['action']} (Avg Confidence: {avg_conf:.2%})"
                        print(f"\r{out_str.ljust(80)}")

                        grouped_results.append(current_group)

                        current_group = {
                            "start": start_time,
                            "end": end_time,
                            "action": predicted_class,
                            "confidences": [confidence]
                        }
            else:
                # If confidence drops below threshold, immediately close and print the current group
                if current_group is not None:
                    avg_conf = sum(current_group["confidences"]) / len(
                        current_group["confidences"])
                    start_str = format_timestamp(current_group["start"])
                    end_str = format_timestamp(current_group["end"])

                    out_str = f"[{start_str} - {end_str}] Action: {current_group['action']} (Avg Confidence: {avg_conf:.2%})"
                    print(f"\r{out_str.ljust(80)}")

                    grouped_results.append(current_group)
                    current_group = None

            # 5. Print progress
            current_end_time = end_frame / fps
            print(
                f"Analyzing... current timestamp: {format_timestamp(current_end_time)}"
                .ljust(80),
                end="\r",
                flush=True)

    except KeyboardInterrupt:
        print("\n\n[!] Analysis interrupted by user. Stopping early...")

    # 6. Print final lingering group
    if current_group is not None:
        avg_conf = sum(current_group["confidences"]) / len(
            current_group["confidences"])
        start_str = format_timestamp(current_group["start"])
        end_str = format_timestamp(current_group["end"])
        out_str = f"[{start_str} - {end_str}] Action: {current_group['action']} (Avg Confidence: {avg_conf:.2%})"
        print(f"\r{out_str.ljust(80)}")
        grouped_results.append(current_group)

    return grouped_results


# ==========================================
# 3. CLI ENTRY POINT
# ==========================================


def main():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    default_labels_path = os.path.join(package_dir, "labels.txt")

    parser = argparse.ArgumentParser(
        description="Analyze video chunks for specific actions.")
    parser.add_argument("video", help="Path to the video file to analyze")
    parser.add_argument(
        "--labels",
        type=str,
        default=default_labels_path,
        help="Path to a text file containing one action label per line")
    parser.add_argument("--chunk_duration",
                        type=float,
                        default=2.0,
                        help="Duration of each chunk in seconds (default: 2.0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (0.0 to 1.0) (default: 0.0)")

    # Example of how you could add a model selector later:
    # parser.add_argument("--model", choices=["xclip", "siglip2"], default="xclip")

    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"Error: The labels file '{args.labels}' was not found.")
        exit(1)

    with open(args.labels, "r") as file:
        my_actions = [line.strip() for line in file if line.strip()]

    if not my_actions:
        print(
            f"Error: The file '{args.labels}' is empty. Please add at least one label."
        )
        exit(1)

    print(f"Loaded {len(my_actions)} labels from '{args.labels}'")

    # --- INSTANTIATE AND LOAD THE MODEL ---
    model = XCLIPModelStrategy()
    model.load()

    # --- RUN THE ANALYSIS ---
    timeline = analyze_video_in_chunks(args.video,
                                       my_actions,
                                       model_strategy=model,
                                       chunk_duration_sec=args.chunk_duration,
                                       threshold=args.threshold)

    if not timeline:
        print("\nNo actions found above the confidence threshold.")
    else:
        print("\nAnalysis complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Script terminated by user.")
        sys.exit(130)
