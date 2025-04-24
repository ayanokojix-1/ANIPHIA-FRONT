import kagglehub

# Download latest version
output = kagglehub.dataset_download("jrobischon/wikipedia-movie-plots",path="./dataset.csv")

print("Path to dataset files:", output)