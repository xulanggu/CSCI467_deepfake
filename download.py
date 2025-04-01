import kagglehub

# Download latest version

path = kagglehub.dataset_download("saurabhbagchi/deepfake-image-detection")

print("Path to dataset files:", path)