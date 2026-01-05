import kagglehub

# Download latest version
path = kagglehub.dataset_download("vijayveersingh/1-2m-brain-signal-data")

print("Path to dataset files:", path)