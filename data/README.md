# HAM10000 Dataset Instructions

This project uses the HAM10000 dataset for skin lesion classification. Due to size and licensing, the full dataset is not included here.

## How to Get the Dataset
1. Go to the official Kaggle page: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Download the ZIP file (requires a free Kaggle account).
3. Unzip the contents into this `/data/` folder. The structure should look like:
   - `/data/ham10000_images_part_1/` (images)
   - `/data/ham10000_images_part_2/` (images)
   - `/data/HAM10000_metadata.csv` (labels and metadata)
   - `/data/HAM10000_segmentations_lesion_tschandl/` (binary masks for lesion isolation)

## Notes
- The notebooks expect images in the above paths—update paths in code if needed.
- Use the binary masks from `/HAM10000_segmentations_lesion_tschandl/` for preprocessing (as described in the paper).
- If you have issues, check Kaggle discussions or the original paper reference [19] for details.

This ensures reproducibility without uploading large files.
