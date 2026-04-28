import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'input_image.jpg'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f'Could not read {image_path}')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img_rgb)
ax.axis('off')
ax.set_title('Click 4 corners for a rectangle, then press n for next or q to finish')

all_rects = []
current = []

while True:
    pts = plt.ginput(n=4, timeout=0, show_clicks=True)
    if len(pts) < 4:
        break
    current = np.array(pts, dtype=float)
    center = current.mean(axis=0)
    all_rects.append({'corners': current, 'center': center})

    xs = np.r_[current[:, 0], current[0, 0]]
    ys = np.r_[current[:, 1], current[0, 1]]
    ax.plot(xs, ys, 'g-')
    ax.plot(center[0], center[1], 'ro')
    ax.text(center[0] + 5, center[1] - 5, f'({center[0]:.1f}, {center[1]:.1f})', color='red')
    fig.canvas.draw_idle()

    key = input('Press n for next rectangle or q to finish: ').strip().lower()
    if key == 'q':
        break

plt.show()

for i, r in enumerate(all_rects, 1):
    c = r['center']
    print(f'Rectangle {i}: center=({c[0]:.2f}, {c[1]:.2f})')
