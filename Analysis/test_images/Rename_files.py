import os

#   Set image folder
folder = 'c:/Users/vagfl/Python Uni/Dissertation/Analysis/test_images/Level_3'
prefix = 'L3_'  #   Change prefix depend of level of clutter folder

# Get all image files
files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

# Rename them
for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]
    new_name = f"{prefix}{i}{ext}"
    os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))

print(f"Renamed {len(files)} files.")