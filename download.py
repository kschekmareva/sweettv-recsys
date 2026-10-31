import kagglehub, os

path = kagglehub.competition_download('sweettv-tv-program-recommender')
print("Path to competition files:", path)
for name in sorted(os.listdir(path)):
    full = os.path.join(path, name)
    print(f"{name:35s} {os.path.getsize(full) / 1e6:8.1f} MB")