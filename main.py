"""
MagicAnimate - Main Script
Simple command-line interface for running MagicAnimate inference
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from demo.animate import MagicAnimate

def main():
    parser = argparse.ArgumentParser(description='MagicAnimate - Human Image Animation')
    parser.add_argument('--mode', type=str, default='gradio', choices=['gradio', 'inference'],
                        help='Run mode: gradio (web UI) or inference (command-line)')
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
    parser.add_argument('--share', action='store_true',
                        help='Create public Gradio link (only for gradio mode)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  WARNING: CUDA not available - running on CPU (very slow)")
        print("   Install CUDA-enabled PyTorch for GPU acceleration:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    if args.mode == 'gradio':
        print("\n" + "="*60)
        print("Starting Gradio Web Interface...")
        print("="*60)
        
        import gradio as gr
        from PIL import Image
        import imageio
        import numpy as np
        
        # Initialize animator
        print("\nInitializing MagicAnimate...")
        animator = MagicAnimate()
        print("✓ Initialization complete!\n")
        
        def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale):
            return animator(reference_image, motion_sequence_state, seed, steps, guidance_scale)
        
        def read_video(video):
            reader = imageio.get_reader(video)
            fps = reader.get_meta_data()['fps']
            return video
        
        def read_image(image, size=512):
            return np.array(Image.fromarray(image).resize((size, size)))
        
        # Create Gradio interface
        with gr.Blocks() as demo:
            gr.HTML("""
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <div>
                        <h1>MagicAnimate: Temporally Consistent Human Image Animation</h1>
                        <p>Animate a reference image using a motion sequence</p>
                    </div>
                </div>
            """)
            
            animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
            
            with gr.Row():
                reference_image = gr.Image(label="Reference Image")
                motion_sequence = gr.Video(format="mp4", label="Motion Sequence")
                
                with gr.Column():
                    random_seed = gr.Textbox(label="Random seed", value="1", info="default: -1")
                    sampling_steps = gr.Textbox(label="Sampling steps", value="25", info="default: 25")
                    guidance_scale = gr.Textbox(label="Guidance scale", value="7.5", info="default: 7.5")
                    submit = gr.Button("Animate")
            
            # Event handlers
            motion_sequence.upload(read_video, motion_sequence, motion_sequence)
            reference_image.upload(read_image, reference_image, reference_image)
            submit.click(animate, 
                        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
                        animation)
            
            # Examples
            gr.Markdown("## Examples")
            gr.Examples(
                examples=[
                    ["inputs/applications/source_image/monalisa.png", "inputs/applications/driving/densepose/running.mp4"],
                    ["inputs/applications/source_image/demo4.png", "inputs/applications/driving/densepose/demo4.mp4"],
                    ["inputs/applications/source_image/dalle2.jpeg", "inputs/applications/driving/densepose/running2.mp4"],
                    ["inputs/applications/source_image/dalle8.jpeg", "inputs/applications/driving/densepose/dancing2.mp4"],
                ],
                inputs=[reference_image, motion_sequence],
                outputs=animation,
            )
        
        print("\n" + "="*60)
        print("Launching Gradio Interface...")
        print("="*60 + "\n")
        demo.launch(share=args.share)
        
    else:  # inference mode
        print("\n" + "="*60)
        print("Running Inference Mode...")
        print("="*60)
        
        from PIL import Image
        import numpy as np
        from magicanimate.utils.videoreader import VideoReader
        
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
        animator = MagicAnimate()
        print("✓ Initialization complete!\n")
        
        # Load inputs
        print("Loading inputs...")
        source_image = np.array(Image.open(args.reference_image).resize((512, 512)))
        motion_sequence = args.motion_sequence
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        
        # Run inference
        print("\n" + "="*60)
        print("Running animation...")
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
        print(f"✓ Output saved to: {output_path}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
