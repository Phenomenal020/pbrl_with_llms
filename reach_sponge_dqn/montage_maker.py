import cv2
import argparse
import sys

class ImageMontage:

    def __init__(self, grid_size_px=768, rows=2, cols=2):
        self.grid_size_px = grid_size_px
        self.rows = rows
        self.cols = cols
        self.cell_width = grid_size_px // cols
        self.cell_height = grid_size_px // rows

    def create_montage(self, img_paths, output_path="enhanced_rl/montage/montage.jpg"):
        # Expecting exactly four images: start/end of two trajectories
        if len(img_paths) != 4:
            raise ValueError(f"Expected 4 images, got {len(img_paths)}")

        # Load and resize to fixed cell size
        imgs = []
        for path in img_paths:
            im = cv2.imread(path)
            if im is None:
                raise FileNotFoundError(f"Could not load image: {path}")
            imgs.append(cv2.resize(im, (self.cell_width, self.cell_height)))

        # Top row: obs1 start and end
        top_row = cv2.hconcat(imgs[0:2])
        # Bottom row: obs2 start and end
        bottom_row = cv2.hconcat(imgs[2:4])
        # Stack rows vertically
        montage = cv2.vconcat([top_row, bottom_row])

        # Save montage image
        cv2.imwrite(output_path, montage)
        # Return both the image array and the path
        return montage, output_path