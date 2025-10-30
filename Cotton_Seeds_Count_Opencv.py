import cv2
import numpy as np

def count_seeds(img_path):
    # Read & convert to grayscale
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    # Modify CLAHE: anything > (75,75,75) → white
    clahe_color = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    mask = (clahe_color[:,:,0] > 45) & (clahe_color[:,:,1] > 45) & (clahe_color[:,:,2] > 45)
    mod_clahe = clahe_color.copy()
    mod_clahe[mask] = [255, 255, 255]

    # Count directly from modified CLAHE (before watershed)
    gray_mod = cv2.cvtColor(mod_clahe, cv2.COLOR_BGR2GRAY)
    _, binary_mod = cv2.threshold(gray_mod, 240, 255, cv2.THRESH_BINARY_INV)
    num_labels_mod, labels_mod = cv2.connectedComponents(binary_mod)
    seed_count_mod = num_labels_mod - 1

    # Watershed to separate all necks (applied on modified CLAHE)
    # More aggressive morphology + lower distance threshold for better splitting
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary = binary_mod.copy()
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.25*dist.max(), 255, 0)  # was 0.35 earlier
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_img = mod_clahe.copy()
    cv2.watershed(ws_img, markers)
    ws_img[markers == -1] = [255, 255, 255]

    # Count from watershed-separated CLAHE
    gray_ws = cv2.cvtColor(ws_img, cv2.COLOR_BGR2GRAY)
    _, binary_ws = cv2.threshold(gray_ws, 240, 255, cv2.THRESH_BINARY_INV)
    num_labels_ws, labels_ws = cv2.connectedComponents(binary_ws)
    seed_count_ws = num_labels_ws - 1

    print(f"🟢 Seed Count (Modified CLAHE): {seed_count_mod}")
    print(f"🔵 Seed Count (Watershed CLAHE): {seed_count_ws}")

    # Display images
    cv2.imshow("Gray", gray)
    cv2.imshow("CLAHE", clahe_img)
    cv2.imshow("Modified CLAHE", gray_mod)
    cv2.imshow("Watershed CLAHE", gray_ws)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return seed_count_mod, seed_count_ws


if __name__ == "__main__":
    path = ""
    count_seeds(path)
