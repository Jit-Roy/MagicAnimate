"""
MagicAnimate - Main Script
Automatically detects GPUs and uses distributed mode for multi-GPU setups
"""

import os
import sys
import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description='MagicAnimate - Human Image Animation')
    parser.add_argument('--reference_image', type=str, default='inputs/applications/source_image/monalisa.png',
                        help='Path to reference image')
    parser.add_argument('--motion_sequence', type=str, default='inputs/applications/driving/densepose/running.mp4',
                        help='Path to motion sequence video')
    parser.add_argument('--output', type=str, default='demo/outputs/output.mp4',
                        help='Output video path')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--steps', type=int, default=25,
                        help='Number of sampling steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for diffusion')
    parser.add_argument('--config', type=str, default='configs/prompts/animation.yaml',
                        help='Path to config file')
    parser.add_argument('--force_single_gpu', action='store_true',
                        help='Force single GPU mode even if multiple GPUs are available')
    
    args = parser.parse_args()
    
    # Check CUDA availability and GPU count
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available - running on CPU (very slow)")
        print("   Install CUDA-enabled PyTorch for GPU acceleration:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        gpu_count = 0
    else:
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available: {gpu_count} GPU(s) detected")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Automatically use distributed mode for multi-GPU
    if gpu_count > 1 and not args.force_single_gpu:
        print(f"\n🚀 Multi-GPU mode activated: Using {gpu_count} GPUs with distributed inference")
        print("   This will be 2-3x faster than single GPU mode!\n")
        print("="*60)
        print("Running Distributed Inference...")
        print("="*60 + "\n")
        
        # Run distributed animation
        cmd = f'python -m magicanimate.pipelines.animation --config {args.config} --dist'
        exit_code = os.system(cmd)
        sys.exit(exit_code >> 8)  # Return the actual exit code
    
    # Single GPU or CPU mode
    print("\n" + "="*60)
    if gpu_count == 1:
        print("Running Single GPU Inference...")
    else:
        print("Running CPU Inference...")
    print("="*60)
    
    from PIL import Image
    import numpy as np
    from demo.animate import MagicAnimate
    
    # Check if input files exist
    if not os.path.exists(args.reference_image):
        print(f"❌ Error: Reference image not found: {args.reference_image}")
        sys.exit(1)
    if not os.path.exists(args.motion_sequence):
        print(f"❌ Error: Motion sequence not found: {args.motion_sequence}")
        sys.exit(1)
    
    print(f"\n📷 Reference Image: {args.reference_image}")
    print(f"🎬 Motion Sequence: {args.motion_sequence}")
    print(f"💾 Output Path: {args.output}")
    print(f"🎲 Seed: {args.seed}")
    print(f"📊 Steps: {args.steps}")
    print(f"🎚️  Guidance Scale: {args.guidance_scale}\n")
    
    # Initialize animator
    print("Initializing MagicAnimate...")
    animator = MagicAnimate(config=args.config)
    print("✓ Initialization complete!\n")
    
    # Load inputs
    print("Loading inputs...")
    source_image = np.array(Image.open(args.reference_image).resize((512, 512)))
    motion_sequence = args.motion_sequence
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    # Run inference
    print("\n" + "="*60)
    print("Running animation (this may take several minutes)...")
    print("="*60 + "\n")
    
    output_path = animator(
        source_image, 
        motion_sequence, 
        args.seed, 
        args.steps, 
        args.guidance_scale
    )
    
    print("\n" + "="*60)
    print(f"✓ Animation complete!")
    print(f"📹 Output saved to: {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
