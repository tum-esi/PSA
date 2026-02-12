import numpy as np
import cv2
import locale
import platform
import sys
import matplotlib.pyplot as plt

GERMAN_SETUP = True if "de_DE" in locale.getlocale()[0] else False
SEP_CHAR = '/' if not "Windows" in platform.platform() else '\\'

CAR_SEM_SEG_COLOR = (0, 0, 143) # RGB
THRESHOLD_LOWER = 0.95
THRESHOLD_UPPER = 0.7

def find_blue_edge(image_path, blue_color=CAR_SEM_SEG_COLOR, tolerance=40):
    """
    Find the top edge of the bottom blue region in a segmentation map.
    Returns a list of (x, y) coordinates marking the edge.
    
    :param image_path: Path to the segmentation map image.
    :param blue_color: RGB tuple for blue (default: CAR_SEM_SEG_COLOR).
    :param tolerance: Allowed color difference for matching blue.
    :return: List of (x, y) coordinates for the edge.
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    height, width, _ = img.shape
    
    # Define blue color range
    lower_blue = np.array([blue_color[0]-tolerance, blue_color[1]-tolerance, blue_color[2]-tolerance])
    upper_blue = np.array([blue_color[0]+tolerance, blue_color[1]+tolerance, blue_color[2]+tolerance])
    
    # Create mask for blue pixels
    mask = cv2.inRange(img, lower_blue, upper_blue)
    
    edge_coords = []
    
    # For each column, find the top edge of blue region
    for x in range(width):
        found_blue = False
        for y in range(height-1, -1, -1):  # bottom to top
            if mask[y, x] > 0:  # Blue pixel
                found_blue = True
            elif found_blue:  # First non-blue after blue
                edge_coords.append((x, y + 1))  # Last blue pixel
                break
    
    return edge_coords

def approximate_with_plot(baseImagePath: str, edge_pixels: list, approximations=[2]):
    img = cv2.imread(baseImagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    height, width, _ = img.shape

    xs = np.asarray([x for x, y in edge_pixels])
    ys = np.asarray([y for x, y in edge_pixels])

    sort_idx = np.argsort(xs)
    xs_sorted = xs[sort_idx]
    ys_sorted = ys[sort_idx]

    plt.imshow(img)
    plt.axis('off')
    if True:
        plt.scatter(xs_sorted, ys_sorted, color='red', s=10, label='Edge Pixels')

    xfit = np.linspace(xs_sorted.min(), xs_sorted.max(), 500)

    for approx in approximations:
        coeffs = np.polyfit(xs_sorted, ys_sorted, approx)
        print(f'{approx}-degree coeffs: {str(coeffs).replace('.', ',')}')
        poly = np.poly1d(coeffs)
        yfit = poly(xfit)
        plt.plot(xfit, yfit, linewidth=2, label=f'{approx}-degree fit', color='red')

        if approx == 2 and coeffs[1] < -1.5e-1:
            print("adversarial image")

    plt.show()



def main(baseImagePath: str):
    edge_pixels = find_blue_edge(baseImagePath)
    print("Number of edge pixels:", len(edge_pixels))
    print("Sample coordinates:", edge_pixels[:10])

    img = cv2.imread(baseImagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    edge_pixels = [(x, y) for (x, y) in edge_pixels if y <= (img.shape[0] * THRESHOLD_LOWER)]
    edge_pixels = [(x, y) for (x, y) in edge_pixels if y > (img.shape[0] * THRESHOLD_UPPER)]
    
    xs = [x for x, y in edge_pixels]
    ys = [y for x, y in edge_pixels]

    approximate_with_plot(baseImagePath, edge_pixels)



if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            main(sys.argv[1])
        else:
            print("Please use the following Syntax: python " + sys.argv[0] + " <sem-seg image>")
    except KeyboardInterrupt:
        pass