import os
import pandas as pd

def prepare_image_data(root_dir, output_csv="image_data.csv"):
    """
    Walk through the directory `root_dir` and find images in subfolders named 'fake' or 'real'.
    Classify them as 'train' or 'test' (or 'sample') depending on the parent folder's name.
    Output a CSV file listing each image path, label, and split.
    """
    data_rows = []

    
    for current_path, dirs, files in os.walk(root_dir):
        if "fake" in current_path.lower():
            label = "fake"
        elif "real" in current_path.lower():
            label = "real"
        else:
            
            continue

    
        if "train" in current_path.lower():
            split = "train"
        elif "test" in current_path.lower():
            split = "test"
        else:
            split = "sample"

        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(current_path, fname)
                data_rows.append({
                    "path": full_path,
                    "label": label,
                    "split": split
                })

    df = pd.DataFrame(data_rows, columns=["path", "label", "split"])

    df.to_csv(output_csv, index=False)
    print(f"Data CSV saved to: {output_csv}")

if __name__ == "__main__":
    base_dir = "/Users/xuwei/Desktop/spring2025/CSCI467/CSCI467_deepfake/467_data/2"  
    prepare_image_data(base_dir, output_csv="deepfake_dataset.csv")