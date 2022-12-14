from PIL import Image
import glob
import os

IMAGE_ID = 1
NUM_CLUSTER = 2
OUTPUT_DIR = f'.\\temp_GIF'

frames = []
imgs = glob.glob((os.path.join(OUTPUT_DIR, "*.png")))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
    
# Save into a GIF file that loops forever
frames[0].save(OUTPUT_DIR + f'\{NUM_CLUSTER}_clusters.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=250, loop=0)