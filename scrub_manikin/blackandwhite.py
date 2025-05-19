import cv2

def to_black_and_white(img_path: str,
                       output_path: str = None,
                       thresh: int = 127,
                       use_otsu: bool = False) -> None:
    """
    Load an image, convert to pure B/W, and (optionally) save.

    Parameters
    ----------
    img_path : str
        Path to your input image.
    output_path : str
        If provided, the binarized image will be saved here.
    thresh : int
        Threshold value in [0..255] for fixed‐threshold mode.
    use_otsu : bool
        If True, ignore `thresh` and use Otsu's method to pick it automatically.
    """
    # 1) Load & gray
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Simple binary threshold
    if use_otsu:
        # Otsu automatically finds the best global threshold
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Fixed threshold: pixels ≥ thresh → white, else → black
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # 3) Show & (maybe) save
    cv2.imshow("Black & White", bw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, bw)

if __name__ == "__main__":
    # Example usage:
    to_black_and_white(
        img_path="assets/Manikin Camera View_screenshot_with_step_02.04.2025.png",
        output_path="assets/Manikin_BW.png",
        thresh=0,       # or pick any 0–255
        use_otsu=True     # try True first, then switch off if under/over-binarizing
    )