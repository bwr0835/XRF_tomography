import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import os

def load_projections(data_dir):
    """
    Load all projection images from a directory.
    Assumes images are TIFF files sorted by angle.
    """
    data_path = Path(data_dir)
    # Get all tiff files
    image_files = sorted([f for f in data_path.glob('*.tif*')])
    
    if not image_files:
        raise ValueError(f"No TIFF files found in {data_dir}")
    
    # Read first image to get dimensions
    first_img = tifffile.imread(image_files[0])
    
    # Initialize array for all projections
    projections = np.zeros((len(image_files), *first_img.shape), dtype=first_img.dtype)
    
    # Load all projections
    for i, img_file in enumerate(image_files):
        projections[i] = tifffile.imread(img_file)
    
    return projections

def create_sinograms(projections):
    """
    Create sinograms from projection data.
    
    Args:
        projections: 3D numpy array of shape (num_angles, height, width)
                    where each projection[i] is a 2D image at angle i
    
    Returns:
        3D array of shape (height, num_angles, width) where each sinograms[i]
        is a sinogram representing a horizontal slice through the sample
    """
    num_angles, height, width = projections.shape
    sinograms = np.zeros((height, num_angles, width), dtype=projections.dtype)
    
    # For each height position
    for h in range(height):
        # Extract the same row from each projection and stack them
        # This creates a single sinogram for this height
        sinograms[h] = np.array([proj[h, :] for proj in projections])
    
    return sinograms

def save_sinograms(sinograms, output_dir):
    """
    Save each sinogram as a separate TIFF file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i in range(sinograms.shape[0]):
        output_file = output_path / f'sinogram_{i:04d}.tiff'
        tifffile.imwrite(output_file, sinograms[i])
        
def plot_example_sinogram(sinogram, index, output_dir):
    """
    Plot and save an example sinogram
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(sinogram, cmap='viridis', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.title(f'Sinogram {index}')
    plt.xlabel('Detector Position (pixels)')
    plt.ylabel('Projection Angle (index)')
    plt.savefig(os.path.join(output_dir, f'sinogram_{index}_visualization.png'))
    plt.close()

def main():
    # Configure these paths according to your data
    input_dir = "projections"  # Directory containing projection images
    output_dir = "sinograms"   # Directory where sinograms will be saved
    
    print("Loading projections...")
    projections = load_projections(input_dir)
    
    print("Creating sinograms...")
    sinograms = create_sinograms(projections)
    
    print("Saving sinograms...")
    save_sinograms(sinograms, output_dir)
    
    # Plot a few example sinograms
    print("Creating example visualizations...")
    for i in [0, sinograms.shape[0]//2, -1]:  # First, middle, and last sinogram
        plot_example_sinogram(sinograms[i], i, output_dir)
    
    print(f"Processing complete. Sinograms saved in {output_dir}")
    print(f"Total number of sinograms created: {sinograms.shape[0]}")

if __name__ == "__main__":
    main() 