"""Facial Landmark Clusters

Defines anatomical clusters of facial landmarks for targeted movement analysis.
Based on MediaPipe Face Mesh 468 landmark model.
"""

FACIAL_CLUSTERS = {
    # Face outline/shape
    'silhouette': [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ],
    
    # Lips
    'lipsUpperOuter': [161, 185, 40, 39, 37, 0, 267, 269, 270, 409, 29],
    'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    
    # Right eye
    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
    'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
    'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
    'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
    'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],
    'rightEyeIris': [473, 474, 475, 476, 477],
    
    # Right eyebrow
    'rightEyebrowUpper': [156, 70, 63, 105, 66, 107, 55, 193],
    'rightEyebrowLower': [35, 124, 46, 53, 52, 65],
    
    # Left eye
    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
    'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
    'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
    'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
    'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],
    'leftEyeIris': [468, 469, 470, 471, 472],
    
    # Left eyebrow
    'leftEyebrowUpper': [383, 300, 293, 334, 296, 336, 285, 417],
    'leftEyebrowLower': [265, 353, 276, 283, 282, 295],
    
    # Nose and cheeks
    'midwayBetweenEyes': [168],
    'noseTip': [1],
    'noseBottom': [2],
    'noseRightCorner': [98],
    'noseLeftCorner': [327],
    'rightCheek': [205],
    'leftCheek': [425]
}

# Higher-level groupings for analysis
CLUSTER_GROUPS = {
    'mouth': [
        'lipsUpperOuter', 'lipsLowerOuter', 
        'lipsUpperInner', 'lipsLowerInner'
    ],
    'right_eye': [
        'rightEyeUpper0', 'rightEyeLower0',
        'rightEyeUpper1', 'rightEyeLower1',
        'rightEyeUpper2', 'rightEyeLower2',
        'rightEyeLower3', 'rightEyeIris'
    ],
    'left_eye': [
        'leftEyeUpper0', 'leftEyeLower0',
        'leftEyeUpper1', 'leftEyeLower1',
        'leftEyeUpper2', 'leftEyeLower2',
        'leftEyeLower3', 'leftEyeIris'
    ],
    'eyebrows': [
        'rightEyebrowUpper', 'rightEyebrowLower',
        'leftEyebrowUpper', 'leftEyebrowLower'
    ],
    'nose': [
        'noseTip', 'noseBottom',
        'noseRightCorner', 'noseLeftCorner'
    ],
    'cheeks': ['rightCheek', 'leftCheek'],
    'face_shape': ['silhouette']
}

# Clusters commonly associated with specific expressions
EXPRESSION_CLUSTERS = {
    'smile': ['mouth', 'cheeks'],
    'frown': ['mouth', 'eyebrows'],
    'surprise': ['mouth', 'eyebrows', 'right_eye', 'left_eye'],
    'squint': ['right_eye', 'left_eye', 'cheeks'],
    'disgust': ['nose', 'mouth'],
    'concentration': ['eyebrows', 'right_eye', 'left_eye']
}


def get_cluster_indices(cluster_name: str) -> list:
    """Get landmark indices for a specific cluster."""
    return FACIAL_CLUSTERS.get(cluster_name, [])


def get_group_indices(group_name: str) -> list:
    """Get all landmark indices for a cluster group."""
    indices = []
    if group_name in CLUSTER_GROUPS:
        for cluster in CLUSTER_GROUPS[group_name]:
            indices.extend(FACIAL_CLUSTERS.get(cluster, []))
    return list(set(indices))  # Remove duplicates


def get_all_cluster_names() -> list:
    """Get list of all available cluster names."""
    return list(FACIAL_CLUSTERS.keys())


def get_all_group_names() -> list:
    """Get list of all available group names."""
    return list(CLUSTER_GROUPS.keys()) 

if __name__ == "__main__":
    print(get_cluster_indices('lipsUpperOuter'))
    print(get_group_indices('mouth'))
    print(get_all_cluster_names())
    print(get_all_group_names())
