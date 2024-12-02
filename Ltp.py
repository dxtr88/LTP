import numpy as np
import cv2
import matplotlib.pyplot as plt



#Computes Local Ternary Patterns for an input image
def compute_ltp(image, k=5, radius=1, points=8):
    """
    Args:
        image: Input grayscale image as numpy array
        k: Threshold constant (default=5)
        radius: Radius of the circular pattern (default=1)
        points: Number of sampling points (default=8)
    """
    rows, cols = image.shape
    upper_pattern = np.zeros_like(image, dtype=np.uint8)
    lower_pattern = np.zeros_like(image, dtype=np.uint8)
    
    # Generate sampling coordinates for circular neighborhood
    angles = 2 * np.pi * np.arange(points) / points
    x_coords = radius * np.cos(angles)
    y_coords = -radius * np.sin(angles)
    
    # Process each pixel in the image
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center_val = image[i, j]
            pattern_upper = 0
            pattern_lower = 0
            
            # Compare with neighbors
            for p in range(points):
                # Calculate neighbor coordinates
                x = j + x_coords[p]
                y = i + y_coords[p]
                
                # Get interpolated pixel value
                x1, x2 = int(np.floor(x)), int(np.ceil(x))
                y1, y2 = int(np.floor(y)), int(np.ceil(y))
                
                if x1 == x2 and y1 == y2:
                    neighbor_val = image[y1, x1]
                else:
                    # Bilinear interpolation
                    f11 = image[y1, x1]
                    f12 = image[y1, x2]
                    f21 = image[y2, x1]
                    f22 = image[y2, x2]
                    
                    x_diff = x - x1
                    y_diff = y - y1
                    
                    neighbor_val = (f11 * (1 - x_diff) * (1 - y_diff) +
                                  f12 * x_diff * (1 - y_diff) +
                                  f21 * (1 - x_diff) * y_diff +
                                  f22 * x_diff * y_diff)
                
                # Apply ternary pattern logic
                if neighbor_val > center_val + k:
                    pattern_upper |= (1 << p)
                elif neighbor_val < center_val - k:
                    pattern_lower |= (1 << p)
                    
            upper_pattern[i, j] = pattern_upper
            lower_pattern[i, j] = pattern_lower
            
    return upper_pattern, lower_pattern


#Compute LTP histogram for the input image.
def get_ltp_histogram(image, k=5, radius=1, points=8):
    """
    Args:
        image: Input grayscale image
        k: Threshold constant
        radius: Radius of circular pattern
        points: Number of sampling points
    """
    upper_pattern, lower_pattern = compute_ltp(image, k, radius, points)
    
    # Calculate histograms
    bins = 2 ** points
    upper_hist = np.histogram(upper_pattern.ravel(), bins=bins, range=(0, bins))[0]
    lower_hist = np.histogram(lower_pattern.ravel(), bins=bins, range=(0, bins))[0]
    
    # Normalize histograms
    upper_hist = upper_hist.astype(float) / upper_hist.sum()
    lower_hist = lower_hist.astype(float) / lower_hist.sum()
    
    # Concatenate histograms
    return np.concatenate([upper_hist, lower_hist]), upper_pattern, lower_pattern

def visualize_results(original, upper_pattern, lower_pattern, hist):
    """
    Visualize the LTP results
    """
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(231)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Upper pattern
    plt.subplot(232)
    plt.imshow(upper_pattern, cmap='gray')
    plt.title('Upper Pattern')
    plt.axis('off')
    
    # Lower pattern
    plt.subplot(233)
    plt.imshow(lower_pattern, cmap='gray')
    plt.title('Lower Pattern')
    plt.axis('off')
    
    # Combined histogram
    plt.subplot(212)
    plt.plot(hist)
    plt.title('LTP Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Normalized Frequency')
    
    plt.tight_layout()
    plt.show()


#Test LTP algorithm on a given image and visualize results.
def test_ltp_on_image(image_path, k=5, radius=1, points=8):
    """
    Args:
        image_path: Path to the input image
        k: Threshold constant (default=5)
        radius: Radius of circular pattern (default=1)
        points: Number of sampling points (default=8)
    """
    # Load and preprocess the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Get LTP histogram and patterns
    ltp_hist, upper_pattern, lower_pattern = get_ltp_histogram(
        gray_image, k=k, radius=radius, points=points
    )
    
    # Visualize results
    visualize_results(gray_image, upper_pattern, lower_pattern, ltp_hist)
    
    print(f"LTP histogram shape: {ltp_hist.shape}")
    print("First few histogram values:", ltp_hist[:10])

# Test the implementation with sample image
# put the path of the image here(replace "Capture.png"): 


test_ltp_on_image('Capture.png') 
