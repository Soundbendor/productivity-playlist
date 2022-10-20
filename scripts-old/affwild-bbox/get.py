import helper, os, sys
import pandas as pd
import numpy as np
from PIL import Image

root = "./data/affwild"
helper.makeDir("{}/cropped".format(root))
vids = os.listdir("{}/frames".format(root))

frameids = []
analysis = {
    "h_old": [],
    "w_old": [],
    "h_box": [],
    "w_box": [],
    "h_new": [],
    "w_new": [],
}

for v in vids:
    print(v)
    helper.makeDir("{}/cropped/{}".format(root, v))
    frames = [s[:-4] for s in os.listdir("{}/frames/{}".format(root, v))]

    for f in frames:
        points = np.array([])
        try:
            points = helper.read_pts("{}/bboxes/train/{}/{}.pts".format(root, v, f))
        except OSError as e:
            print("Could not find point file for vid {} frame {}".format(v, f), file=sys.stderr)
            continue

        left, top, right, bottom = helper.process_bbox(points)
        h_box, w_box = int(bottom) - int(top), int(right) - int(left)

        full = Image.open("{}/frames/{}/{}.png".format(root, v, f))
        h_old, w_old = full.size
            
        crop = full.crop((left, top, right, bottom))
        h_new, w_new = crop.size
        crop.save("{}/cropped/{}/{}.png".format(root, v, f))

        frameids.append(int(1e5 * int(v) + int(f)))
        analysis["h_box"].append(h_box)
        analysis["w_box"].append(w_box)
        analysis["h_old"].append(h_old)
        analysis["w_old"].append(w_old)
        analysis["h_new"].append(h_new)
        analysis["w_new"].append(w_new)

pd.DataFrame(analysis, index=frameids).to_csv(path_or_buf="affwild_bbox_data.csv")
