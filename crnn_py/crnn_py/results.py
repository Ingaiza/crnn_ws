#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from my_custom_interfaces.msg import ResultsData

import os
import sys
import numpy as np
import torch
import soundfile as sf
import torchaudio.transforms as T
import torch.nn.functional as F
import xgboost as xgb
from ament_index_python.packages import get_package_share_directory
import itertools # Required for truth table generation

# --- ALIASING UTILS FOR PICKLE ---
import sys
from . import utils
sys.modules["utils"] = utils
sys.modules["utils.logger"] = utils.logger
sys.modules["utils.util"] = utils.util


from .net.model import AudioCRNN

TEST_FILE_PATH = "/home/ingaiza/CRNN/dataset/audio/val/chainsaw10s.wav"
THRESHOLD_SAMPLES = 240000 

class AudioProcessor(Node): 
    def __init__(self):
        super().__init__("Processor")
        self.publisher_ = self.create_publisher(ResultsData, "audio_results", 10)
        self.get_logger().info("Audio Processor Node Started.")

        pkg_share_dir = get_package_share_directory('crnn_py')
        models_dir = os.path.join(pkg_share_dir, 'models')

        self.CRNN_MODEL_PATH = os.path.join(models_dir, 'model_best.pth')
        self.XGB_MODEL_PATH = os.path.join(models_dir, 'xgboost_audio_classifier.json')
        self.CFG_PATH = os.path.join(models_dir, 'crnn.cfg')

        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_map_ = {0: "Natural", 1: "Unnatural", 2: "Human Sound"}

        self.crnn_ = None
        self.xgb_ = None
        self.load_models()

        # Validation Parameters
        self.ACCURACY_BASE = 0.76
        self.WEIGHT_LOW = self.ACCURACY_BASE          # 0.76
        self.WEIGHT_HIGH = 1.0 + (1.0 - self.ACCURACY_BASE) # 1.24

        self.timer_ = self.create_timer(20.0, self.process_audio_file)
        self.get_logger().info(f"Node ready. Processing '{os.path.basename(TEST_FILE_PATH)}' every 20s.")

    def load_models(self):
        self.get_logger().info("Loading Models...")
        config = {'cfg': self.CFG_PATH, 'transforms': {'args': {'channels': 'mono'}}}
        self.crnn_ = AudioCRNN(classes=self.class_map_.values(), config=config)
        
        try:
            checkpoint = torch.load(self.CRNN_MODEL_PATH, map_location=self.device_, weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                self.crnn_.load_state_dict(state_dict)
            else:
                self.crnn_.load_state_dict(checkpoint)
        except Exception as e:
            self.get_logger().error(f"Error loading CRNN: {e}")
            return
        
        self.crnn_.eval().to(self.device_)
        self.xgb_ = xgb.XGBClassifier()
        self.xgb_.load_model(self.XGB_MODEL_PATH)
        self.get_logger().info("Models Loaded.")

    def validate_prediction(self, probs):
        """
        Implements the Truth Table Sensitivity Analysis.
        Input: probs (list/array of 3 floats) [Natural, Unnatural, Human]
        Output: Boolean (True if valid, False if weak)
        """
        original_winner_idx = np.argmax(probs)
        original_winner_class = self.class_map_[original_winner_idx]
        
        # 1. Generate Truth Table of Multipliers (0.76, 1.24)
        # Cartesian product of [Low, High] for 3 classes = 8 combinations
        multipliers = [self.WEIGHT_LOW, self.WEIGHT_HIGH]
        combinations = list(itertools.product(multipliers, repeat=3))
        
        wins = 0
        total_scenarios = len(combinations)

        # Debug logs to see the table
        # self.get_logger().info(f"--- VALIDATION TABLE (Winner: {original_winner_class}) ---")
        
        for i, coeffs in enumerate(combinations):
            # Apply weights: [P_N * w1,  P_U * w2,  P_H * w3]
            weighted_probs = np.array(probs) * np.array(coeffs)
            
            # Who wins this round?
            round_winner_idx = np.argmax(weighted_probs)
            
            if round_winner_idx == original_winner_idx:
                wins += 1
            
            # Optional: Log "close calls" where it fails
            # else:
            #    self.get_logger().info(f"   Failed Row {i}: Weights {coeffs} -> Winner {self.class_map_[round_winner_idx]}")

        pass_ratio = wins / total_scenarios
        is_valid = wins > (total_scenarios / 2) # Majority (>4 out of 8)

        self.get_logger().info(f"Validation: {original_winner_class} won {wins}/{total_scenarios} scenarios ({pass_ratio:.0%})")
        
        if not is_valid:
            self.get_logger().warn(f"ALERT BLOCKED: Prediction '{original_winner_class}' too weak (failed sensitivity check).")
            
        return is_valid

    def process_audio_file(self):
        if self.crnn_ is None or self.xgb_ is None: return

        self.get_logger().info(f"Processing: {TEST_FILE_PATH}")
        try:
            data, sr = sf.read(TEST_FILE_PATH)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.permute(1, 0)
            if sr != 16000: waveform = T.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            num_samples = waveform.shape[1]
            final_probs = None
            
            # --- Inference Logic (Adaptive) ---
            if num_samples > THRESHOLD_SAMPLES:
                seqs = waveform.permute(1, 0).unsqueeze(0)
                batch = self.create_batch(seqs)
                with torch.no_grad():
                    feat = self.crnn_(batch, return_features=True).cpu().numpy()
                    final_probs = self.xgb_.predict_proba(feat)[0]
            else:
                WINDOW_SIZE, STRIDE = 96000, 48000
                windows = []
                if num_samples < WINDOW_SIZE:
                    chunk = F.pad(waveform, (0, WINDOW_SIZE - num_samples))
                    windows.append(chunk)
                else:
                    for i in range(0, num_samples, STRIDE):
                        chunk = waveform[:, i : i + WINDOW_SIZE]
                        if chunk.shape[1] < 8000: continue
                        if chunk.shape[1] < WINDOW_SIZE:
                            chunk = F.pad(chunk, (0, WINDOW_SIZE - chunk.shape[1]))
                        windows.append(chunk)
                
                probs_list = []
                for w in windows:
                    seqs = w.permute(1, 0).unsqueeze(0)
                    batch = self.create_batch(seqs)
                    with torch.no_grad():
                        feat = self.crnn_(batch, return_features=True).cpu().numpy()
                        probs_list.append(self.xgb_.predict_proba(feat)[0])
                final_probs = np.mean(probs_list, axis=0)

            # --- NEW VALIDATION STEP ---
            if self.validate_prediction(final_probs):
                # Only publish if Valid
                msg = ResultsData()
                msg.natural = int(final_probs[0] * 100)
                msg.unnatural = int(final_probs[1] * 100)
                msg.human = int(final_probs[2] * 100)
                self.publisher_.publish(msg)
                self.get_logger().info(f"PUBLISHED: [N:{msg.natural} U:{msg.unnatural} H:{msg.human}]")
            else:
                # If invalid, you might want to publish "Unknown" or nothing
                self.get_logger().info("Result discarded by Validation Layer.")

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    def create_batch(self, seqs):
        lengths = torch.tensor([seqs.shape[1]]).long()
        srs = torch.tensor([16000]).long()
        return (seqs.to(self.device_), lengths.to(self.device_), srs.to(self.device_))

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessor() 
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()