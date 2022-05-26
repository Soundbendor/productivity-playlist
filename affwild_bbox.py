import helper, os
import pandas as pd
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
        try:
            left, top, right, bottom = helper.makeBBox(helper.read_pts("{}/bboxes/train/{}/{}.pts".format(root, v, f)))
            h_box, w_box = int(bottom) - int(top), int(right) - int(left)

            imag = Image.open("{}/frames/{}/{}.png".format(root, v, f))
            h_old, w_old = imag.size
            
            crop = Image.crop((left, top, right, bottom))
            h_new, w_new = crop.size
            crop.save("{}/cropped/{}/{}.png".format(root, v, f))

            frameids.append(int(1e5 * v + f))
            analysis["h_box"].append(h_box)
            analysis["w_box"].append(w_box)
            analysis["h_old"].append(h_old)
            analysis["w_old"].append(w_old)
            analysis["h_new"].append(h_new)
            analysis["w_new"].append(w_new)
        except:
            print("Could not successfully process frame {}".format(f))

pd.DataFrame(analysis, index=frameids).to_csv(path_or_buf="affwild_bbox_data.csv")