from montage_maker import ImageMontage
import cv2

obsdir = f"enhanced_rl/observations1"

mm = ImageMontage(grid_size_px=768, rows=2, cols=2)
montage, _ = mm.create_montage([f"{obsdir}/20250514_040841_9346323000_1.png", f"{obsdir}/20250514_032856_8125153000_1.png", f"{obsdir}/20250514_041329_9640783000_1.png", f"{obsdir}/20250514_041208_5945333000_1.png"], output_path="enhanced_rl/montage/test.png")

# cv2.imshow("Montage", montage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()