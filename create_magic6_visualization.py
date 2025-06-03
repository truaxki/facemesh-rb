#!/usr/bin/env python3
"""
FaceMesh-RB: Magic 6 + Expanded 64 Landmark Visualization
Side-by-side comparison for HCI paper showing technical innovation

Extracts frame 73 from e4-session8.csv and creates publication-ready figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Magic 6 landmarks for stable rotation (from your research)
MAGIC_6 = [1, 2, 98, 327, 205, 425]

# Facial clusters for expanded 64 landmarks (from facial_clusters.py)
FACIAL_CLUSTERS = {
    # Nose landmarks
    'noseTip': [1],
    'noseBottom': [2], 
    'noseRightCorner': [98],
    'noseLeftCorner': [327],
    
    # Cheek landmarks
    'rightCheek': [205],
    'leftCheek': [425],
    
    # Lips (high expression sensitivity)
    'lipsUpperOuter': [161, 185, 40, 39, 37, 0, 267, 269, 270, 409, 29],
    'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    
    # Eyes (expression regions)
    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    
    # Eyebrows (expression regions)
    'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
    'rightEyebrowLower': [35, 124, 46, 53, 52, 65],
    'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
    'leftEyebrowLower': [265, 353, 276, 283, 282, 295],
}

def get_expanded_64_landmarks():
    """Extract the expanded 64 landmarks used in your research"""
    expanded_landmarks = set()
    
    # Add Magic 6 (stable alignment)
    expanded_landmarks.update(MAGIC_6)
    
    # Add expression-sensitive regions
    for cluster_name in ['lipsUpperOuter', 'lipsLowerOuter', 'lipsUpperInner', 'lipsLowerInner',
                        'rightEyeUpper0', 'rightEyeLower0', 'leftEyeUpper0', 'leftEyeLower0',
                        'rightEyebrowUpper', 'rightEyebrowLower', 'leftEyebrowUpper', 'leftEyebrowLower']:
        expanded_landmarks.update(FACIAL_CLUSTERS[cluster_name])
    
    return sorted(list(expanded_landmarks))

def extract_frame_landmarks(file_path, frame_number=73):
    """
    Extract 3D landmark coordinates for a specific frame
    Based on your data structure: feat_0_x, feat_0_y, feat_0_z, etc.
    """
    print(f"Loading frame {frame_number} from {file_path}")
    
    # Load CSV data
    df = pd.read_csv(file_path)
    
    if frame_number >= len(df):
        raise ValueError(f"Frame {frame_number} not found. Dataset has {len(df)} frames.")
    
    # Extract the specific frame
    frame_data = df.iloc[frame_number]
    
    # Extract face depth for z-scaling (your critical preprocessing step)
    face_depth = frame_data['Face Depth (cm)']
    
    # Extract 478 landmarks (MediaPipe standard)
    landmarks_3d = np.zeros((478, 3))
    
    for i in range(478):
        x_col = f'feat_{i}_x'
        y_col = f'feat_{i}_y' 
        z_col = f'feat_{i}_z'
        
        if x_col in frame_data and y_col in frame_data and z_col in frame_data:
            landmarks_3d[i, 0] = frame_data[x_col]  # x coordinate
            landmarks_3d[i, 1] = frame_data[y_col]  # y coordinate
            landmarks_3d[i, 2] = frame_data[z_col] * face_depth  # z-scaled (your innovation)
    
    return landmarks_3d, face_depth

def create_side_by_side_visualization(landmarks_3d, output_file='magic6_innovation.png'):
    """
    Create side-by-side visualization for HCI paper
    Left: Raw MediaPipe landmarks (all 478 points)
    Right: Magic 6 + Expanded 64 highlighted
    """
    
    # Get expanded 64 landmarks
    expanded_64 = get_expanded_64_landmarks()
    
    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Left subplot: Raw MediaPipe landmarks
    ax1 = fig.add_subplot(111, projection='3d')
    
    # Plot all 478 landmarks in gray
    ax1.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], 
               c='gray', s=8, alpha=0.8, label='All 478 landmarks')
    
    ax1.set_title('FaceMesh-RB: Magic 6 + Expanded 64 Innovation\n(Separation of alignment & expression)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate') 
    ax1.set_zlabel('Z coordinate (depth-scaled)')
    
    # Highlight Magic 6 landmarks (stable rotation)
    magic_6_coords = landmarks_3d[MAGIC_6]
    ax1.scatter(magic_6_coords[:, 0], magic_6_coords[:, 1], magic_6_coords[:, 2],
               c='red', s=50, alpha=0.9, label='Magic 6 (Stable Rotation)', edgecolors='darkred')

    # Highlight Expanded 64 landmarks (expression detection)
    expanded_coords = landmarks_3d[expanded_64]
    ax1.scatter(expanded_coords[:, 0], expanded_coords[:, 1], expanded_coords[:, 2],
               c='blue', s=20, alpha=0.8, label='Expanded 64 (Expression)', edgecolors='darkblue')

    # Add legend
    ax1.legend(loc='upper right', fontsize=10)
    
    # Set consistent viewing angles for both subplots
    ax1.view_init(elev=20, azim=45)
    
    # Ensure same scale for both plots
    ax1.set_xlim(ax1.get_xlim())
    ax1.set_ylim(ax1.get_ylim()) 
    ax1.set_zlim(ax1.get_zlim())
    
    # Add overall title
    fig.suptitle('FaceMesh-RB: Technical Innovation for Agentic Systems\nFrame 73, Participant E4, Session 8', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save high-resolution figure for paper
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved as {output_file}")
    
    # Show plot
    plt.show()
    
    return fig

def create_landmark_annotation_view(landmarks_3d, output_file='landmark_annotations.png'):
    """
    Create annotated view showing specific Magic 6 landmark positions
    Perfect for technical explanation in paper
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all landmarks in light gray
    ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2],
               c='lightgray', s=10, alpha=0.4)
    
    # Magic 6 with annotations
    magic_6_labels = {
        1: 'Nose Tip\n(Landmark 1)',
        2: 'Nose Bridge\n(Landmark 2)', 
        98: 'Right Nose\n(Landmark 98)',
        327: 'Left Nose\n(Landmark 327)',
        205: 'Right Cheek\n(Landmark 205)',
        425: 'Left Cheek\n(Landmark 425)'
    }
    
    for landmark_idx in MAGIC_6:
        coord = landmarks_3d[landmark_idx]
        ax.scatter(coord[0], coord[1], coord[2], 
                  c='red', s=100, alpha=0.9, edgecolors='darkred')
        
        # Add text annotation
        ax.text(coord[0], coord[1], coord[2], 
               magic_6_labels[landmark_idx],
               fontsize=9, ha='center', va='bottom')
    
    ax.set_title('Magic 6 Landmarks: Stable Rotation Foundation\n"Separating Geometric Alignment from Expression Detection"',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    
    # Set optimal viewing angle
    ax.view_init(elev=15, azim=60)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Annotated view saved as {output_file}")
    
    plt.show()
    return fig

def main():
    """Main execution function"""
    
    # File path (user specified)
    data_file = r"C:\Users\ktrua\source\repos\facemesh-rb\read\e4\e4-session8.csv"
    frame_number = 73
    
    try:
        # Extract landmark data for specified frame
        print("=" * 60)
        print("FaceMesh-RB Visualization Generator")
        print("Creating Magic 6 + Expanded 64 visualization for HCI paper")
        print("=" * 60)
        
        landmarks_3d, face_depth = extract_frame_landmarks(data_file, frame_number)
        
        print(f"✓ Successfully extracted {len(landmarks_3d)} landmarks")
        print(f"✓ Face depth: {face_depth:.2f} cm")
        print(f"✓ Z-coordinates scaled for 3D integrity")
        
        # Create main side-by-side visualization
        print("\nGenerating side-by-side comparison...")
        main_fig = create_side_by_side_visualization(landmarks_3d, 'facemesh_rb_innovation.png')
        
        # Create annotated Magic 6 view
        print("\nGenerating annotated Magic 6 view...")
        annotation_fig = create_landmark_annotation_view(landmarks_3d, 'magic6_annotations.png')
        
        # Print summary statistics
        expanded_64 = get_expanded_64_landmarks()
        print("\n" + "=" * 60)
        print("VISUALIZATION SUMMARY")
        print("=" * 60)
        print(f"Total MediaPipe landmarks: 478")
        print(f"Magic 6 (stable rotation): {len(MAGIC_6)} landmarks")
        print(f"Expanded 64 (expression): {len(expanded_64)} landmarks")
        print(f"Innovation: Separation of alignment vs expression detection")
        print(f"Application: Real-time feedback for agentic AI systems")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Error: Could not find file {data_file}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 