"""Data Filters Module

Provides Kabsch and Kabsch-Umeyama algorithms for rigid body alignment.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.linalg import svd, det


class DataFilters:
    """Collection of Kabsch and Kabsch-Umeyama alignment algorithms."""
    
    @staticmethod
    def kabsch_algorithm(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kabsch algorithm for optimal rotation matrix calculation.
        
        Finds the optimal rotation matrix that minimizes RMSD between two point sets.
        Based on: https://en.wikipedia.org/wiki/Kabsch_algorithm
        
        Args:
            P: Reference point set (N x 3)
            Q: Point set to align to P (N x 3)
            
        Returns:
            R: Optimal rotation matrix (3 x 3)
            t: Translation vector (3,)
            rmsd: Root mean square deviation after alignment
        """
        assert P.shape == Q.shape, "Point sets must have same shape"
        assert P.shape[1] == 3, "Points must be 3D"
        
        # Step 1: Center both point sets (translation)
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # Step 2: Compute cross-covariance matrix H
        H = P_centered.T @ Q_centered
        
        # Step 3: Singular Value Decomposition
        U, S, Vt = svd(H)
        
        # Step 4: Compute rotation matrix
        # Check for reflection case
        d = det(U @ Vt)
        
        if d < 0:
            # Reflection case - flip the last column of U
            U[:, -1] *= -1
        
        R = U @ Vt
        
        # Step 5: Compute translation
        t = centroid_P - R @ centroid_Q
        
        # Step 6: Calculate RMSD
        Q_aligned = (R @ Q_centered.T).T + centroid_P
        rmsd = np.sqrt(np.mean(np.sum((P - Q_aligned)**2, axis=1)))
        
        return R, t, rmsd
    
    @staticmethod
    def apply_transformation(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Apply rotation and translation to point cloud.
        
        Args:
            points: Point cloud (N x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3,)
            
        Returns:
            Transformed points (N x 3)
        """
        return (R @ points.T).T + t
    
    @staticmethod
    def align_frames_to_baseline(frames_data: List[Dict], baseline_frame_count: int = 30) -> List[Dict]:
        """
        Align all frames to a baseline computed from the average of the first N frames using Kabsch algorithm.
        
        Args:
            frames_data: List of frame dictionaries with 'points' and 'colors'
            baseline_frame_count: Number of initial frames to average for baseline (default: 30)
            
        Returns:
            List of aligned frame dictionaries
        """
        if not frames_data:
            raise ValueError("Empty frames data")
        
        # Determine actual number of frames to use for baseline
        actual_baseline_count = min(baseline_frame_count, len(frames_data))
        
        print(f"🎯 Computing baseline from average of first {actual_baseline_count} frames")
        print(f"📊 Total frames to align: {len(frames_data)}")
        
        # Compute average baseline points
        baseline_frames = frames_data[:actual_baseline_count]
        first_frame_point_count = len(baseline_frames[0]['points'])
        
        # Check that all baseline frames have the same number of points
        for i, frame in enumerate(baseline_frames):
            if len(frame['points']) != first_frame_point_count:
                print(f"⚠️ Baseline frame {i}: Point count mismatch ({len(frame['points'])} vs {first_frame_point_count})")
                raise ValueError(f"Inconsistent point counts in baseline frames")
        
        # Calculate average points across baseline frames
        baseline_points = np.zeros((first_frame_point_count, 3))
        for frame in baseline_frames:
            baseline_points += frame['points']
        baseline_points /= actual_baseline_count
        
        print(f"📍 Baseline computed from {actual_baseline_count} frames with {len(baseline_points)} points each")
        
        aligned_frames = []
        alignment_stats = []
        
        for i, frame_data in enumerate(frames_data):
            current_points = frame_data['points'].copy()
            
            if len(current_points) != len(baseline_points):
                print(f"⚠️ Frame {i}: Point count mismatch ({len(current_points)} vs {len(baseline_points)})")
                # For now, skip frames with different point counts
                # In future, could implement point correspondence matching
                aligned_frames.append(frame_data.copy())
                continue
            
            if i < actual_baseline_count:
                # For frames used in baseline computation, align to the computed average baseline
                R, t, rmsd = DataFilters.kabsch_algorithm(baseline_points, current_points)
                
                # Transform points
                aligned_points = DataFilters.apply_transformation(current_points, R, t)
                
                # Create aligned frame
                aligned_frame = frame_data.copy()
                aligned_frame['points'] = aligned_points
                
                # Store transformation info
                aligned_frame['kabsch_transform'] = {
                    'rotation_matrix': R,
                    'translation_vector': t,
                    'rmsd': rmsd,
                    'baseline_type': f'average_of_{actual_baseline_count}_frames',
                    'is_baseline_frame': True
                }
            else:
                # Apply Kabsch alignment to computed baseline
                R, t, rmsd = DataFilters.kabsch_algorithm(baseline_points, current_points)
                
                # Transform points
                aligned_points = DataFilters.apply_transformation(current_points, R, t)
                
                # Create aligned frame
                aligned_frame = frame_data.copy()
                aligned_frame['points'] = aligned_points
                
                # Store transformation info
                aligned_frame['kabsch_transform'] = {
                    'rotation_matrix': R,
                    'translation_vector': t,
                    'rmsd': rmsd,
                    'baseline_type': f'average_of_{actual_baseline_count}_frames',
                    'is_baseline_frame': False
                }
            
            alignment_stats.append({
                'frame_idx': i,
                'rmsd': rmsd,
                'is_baseline_contributor': i < actual_baseline_count
            })
            
            aligned_frames.append(aligned_frame)
        
        # Print alignment statistics
        baseline_rmsds = [stat['rmsd'] for stat in alignment_stats if stat['is_baseline_contributor']]
        other_rmsds = [stat['rmsd'] for stat in alignment_stats if not stat['is_baseline_contributor']]
        
        if baseline_rmsds:
            print(f"📈 Baseline frames alignment RMSD statistics:")
            print(f"   Mean: {np.mean(baseline_rmsds):.4f}")
            print(f"   Std:  {np.std(baseline_rmsds):.4f}")
            print(f"   Min:  {np.min(baseline_rmsds):.4f}")
            print(f"   Max:  {np.max(baseline_rmsds):.4f}")
        
        if other_rmsds:
            print(f"📈 Non-baseline frames alignment RMSD statistics:")
            print(f"   Mean: {np.mean(other_rmsds):.4f}")
            print(f"   Std:  {np.std(other_rmsds):.4f}")
            print(f"   Min:  {np.min(other_rmsds):.4f}")
            print(f"   Max:  {np.max(other_rmsds):.4f}")
        
        print(f"✅ Kabsch alignment complete using average baseline!")
        return aligned_frames
    
    @staticmethod
    def kabsch_umeyama_algorithm(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Kabsch-Umeyama algorithm for optimal rotation, translation, and scaling.
        
        Finds the optimal similarity transformation (rotation + translation + uniform scaling)
        that minimizes RMSD between two point sets.
        
        Args:
            P: Reference point set (N x 3)
            Q: Point set to align to P (N x 3)
            
        Returns:
            R: Optimal rotation matrix (3 x 3)
            t: Translation vector (3,)
            s: Scaling factor
            rmsd: Root mean square deviation after alignment
        """
        assert P.shape == Q.shape, "Point sets must have same shape"
        assert P.shape[1] == 3, "Points must be 3D"
        
        # Step 1: Center both point sets
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        # Step 2: Compute variance of Q (source points being transformed)
        var_Q = np.sum(Q_centered**2) / len(Q)
        
        # Handle degenerate case
        if var_Q < 1e-10:
            # If Q has no variance, return identity transformation
            R = np.eye(3)
            t = centroid_P - centroid_Q
            s = 1.0
            rmsd = np.sqrt(np.mean(np.sum((P - Q - t)**2, axis=1)))
            return R, t, s, rmsd
        
        # Step 3: Compute cross-covariance matrix (P^T × Q)
        H = P_centered.T @ Q_centered / len(P)
        
        # Step 4: Singular Value Decomposition
        U, S, Vt = svd(H)
        
        # Step 5: Compute rotation matrix with proper reflection handling
        R = U @ Vt
        
        # Check for reflection case
        d = det(R)
        if d < 0:
            # Reflection case - flip the last column of U
            U_corrected = U.copy()
            U_corrected[:, -1] *= -1
            R = U_corrected @ Vt
            # Also flip the corresponding singular value
            S_corrected = S.copy()
            S_corrected[-1] *= -1
        else:
            S_corrected = S
        
        # Step 6: Compute optimal scaling factor
        s = np.sum(S_corrected) / var_Q
        
        # Step 7: Compute translation
        t = centroid_P - s * R @ centroid_Q
        
        # Step 8: Calculate RMSD
        Q_aligned = s * (R @ Q_centered.T).T + centroid_P
        rmsd = np.sqrt(np.mean(np.sum((P - Q_aligned)**2, axis=1)))
        
        return R, t, s, rmsd
    
    @staticmethod
    def apply_transformation_with_scale(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
        """
        Apply rotation, translation, and scaling to point cloud.
        
        Args:
            points: Point cloud (N x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3,)
            s: Scaling factor
            
        Returns:
            Transformed points (N x 3)
        """
        return s * (R @ points.T).T + t
    
    @staticmethod
    def align_frames_to_baseline_umeyama(frames_data: List[Dict], baseline_frame_count: int = 30) -> List[Dict]:
        """
        Align all frames using Kabsch-Umeyama algorithm (includes scaling).
        
        Args:
            frames_data: List of frame dictionaries with 'points' and 'colors'
            baseline_frame_count: Number of initial frames to average for baseline (default: 30)
            
        Returns:
            List of aligned frame dictionaries
        """
        if not frames_data:
            raise ValueError("Empty frames data")
        
        # Determine actual number of frames to use for baseline
        actual_baseline_count = min(baseline_frame_count, len(frames_data))
        
        print(f"🎯 Computing baseline from average of first {actual_baseline_count} frames (Kabsch-Umeyama)")
        print(f"📊 Total frames to align: {len(frames_data)}")
        
        # Compute average baseline points
        baseline_frames = frames_data[:actual_baseline_count]
        first_frame_point_count = len(baseline_frames[0]['points'])
        
        # Check that all baseline frames have the same number of points
        for i, frame in enumerate(baseline_frames):
            if len(frame['points']) != first_frame_point_count:
                print(f"⚠️ Baseline frame {i}: Point count mismatch ({len(frame['points'])} vs {first_frame_point_count})")
                raise ValueError(f"Inconsistent point counts in baseline frames")
        
        # Calculate average points across baseline frames
        baseline_points = np.zeros((first_frame_point_count, 3))
        for frame in baseline_frames:
            baseline_points += frame['points']
        baseline_points /= actual_baseline_count
        
        print(f"📍 Baseline computed from {actual_baseline_count} frames with {len(baseline_points)} points each")
        
        aligned_frames = []
        alignment_stats = []
        
        for i, frame_data in enumerate(frames_data):
            current_points = frame_data['points'].copy()
            
            if len(current_points) != len(baseline_points):
                print(f"⚠️ Frame {i}: Point count mismatch ({len(current_points)} vs {len(baseline_points)})")
                # Skip frames with different point counts
                aligned_frames.append(frame_data.copy())
                continue
            
            # Apply Kabsch-Umeyama alignment
            R, t, s, rmsd = DataFilters.kabsch_umeyama_algorithm(baseline_points, current_points)
            
            # Transform points
            aligned_points = DataFilters.apply_transformation_with_scale(current_points, R, t, s)
            
            # Create aligned frame
            aligned_frame = frame_data.copy()
            aligned_frame['points'] = aligned_points
            
            # Store transformation info
            aligned_frame['kabsch_umeyama_transform'] = {
                'rotation_matrix': R,
                'translation_vector': t,
                'scale_factor': s,
                'rmsd': rmsd,
                'baseline_type': f'average_of_{actual_baseline_count}_frames',
                'is_baseline_frame': i < actual_baseline_count
            }
            
            alignment_stats.append({
                'frame_idx': i,
                'rmsd': rmsd,
                'scale_factor': s,
                'is_baseline_contributor': i < actual_baseline_count
            })
            
            aligned_frames.append(aligned_frame)
        
        # Print alignment statistics
        baseline_stats = [stat for stat in alignment_stats if stat['is_baseline_contributor']]
        other_stats = [stat for stat in alignment_stats if not stat['is_baseline_contributor']]
        
        if baseline_stats:
            print(f"📈 Baseline frames alignment statistics:")
            print(f"   RMSD - Mean: {np.mean([s['rmsd'] for s in baseline_stats]):.4f}")
            print(f"   Scale - Mean: {np.mean([s['scale_factor'] for s in baseline_stats]):.4f}")
        
        if other_stats:
            print(f"📈 Non-baseline frames alignment statistics:")
            print(f"   RMSD - Mean: {np.mean([s['rmsd'] for s in other_stats]):.4f}")
            print(f"   Scale - Mean: {np.mean([s['scale_factor'] for s in other_stats]):.4f}")
        
        print(f"✅ Kabsch-Umeyama alignment complete!")
        return aligned_frames 