from PIL import Image
import os

# ── Config ──────────────────────────────────────────
GIF_PATH   = "figures/trajectory_animation.gif"
OUTPUT_DIR = "figures/keyframes"
NUM_FRAMES = 5   # how many keyframes to extract

os.makedirs(OUTPUT_DIR, exist_ok=True)

gif = Image.open(GIF_PATH)
total_frames = gif.n_frames
print(f"Total frames in GIF: {total_frames}")

# Pick evenly spaced frame indices
indices = [int(i * (total_frames - 1) / (NUM_FRAMES - 1)) 
           for i in range(NUM_FRAMES)]

print(f"Extracting frames at indices: {indices}")

for rank, idx in enumerate(indices):
    gif.seek(idx)
    frame = gif.convert("RGB")
    out_path = os.path.join(OUTPUT_DIR, f"keyframe_{rank+1}.png")
    frame.save(out_path, "PNG")
    print(f"  Saved frame {idx:3d}  →  {out_path}")

print("Done.")