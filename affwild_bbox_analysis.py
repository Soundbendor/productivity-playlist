import pandas as pd
import numpy as np
import helper
import json
import sys

size = int(sys.argv[1])

boxdata = pd.read_csv("./affwild_bbox_data.csv", header=0)
boxlen = int(sys.argv[2]) if len(sys.argv) > 2 else len(boxdata)
errdata = pd.read_csv("./affwild_bbox_err.csv", header=0)
errlen = int(sys.argv[3]) if len(sys.argv) > 3 else len(errdata)

vidids = []
data_by_video = {
    # "square": [],
    # "nonsquare": [],
    "found": [],
    # "nonfound": [],
    # "worked": [],
    # "nonworked": []
}

squarefreqs = {}

print("boxdata")
for i in range(boxlen):
    print(boxdata.iloc[i]["id"])
    vidid = int(boxdata.iloc[i]["id"] // 1e5)
    h_old = boxdata.iloc[i]["h_old"]
    w_old = boxdata.iloc[i]["w_old"]
    h_box = boxdata.iloc[i]["h_box"]
    w_box = boxdata.iloc[i]["w_box"]
    h_new = boxdata.iloc[i]["h_new"]
    w_new = boxdata.iloc[i]["w_new"]

    square = h_new == w_new
    worked = (h_new == h_box) and (w_new == w_box)

    if square and int(h_new) == size:
        if vidid not in vidids:
            vidids.append(vidid)
            # data_by_video["square"].append(1 if square else 0)
            # data_by_video["nonsquare"].append(0 if square else 1)
            # data_by_video["worked"].append(1 if worked else 0)
            # data_by_video["nonworked"].append(0 if worked else 1)
            data_by_video["found"].append(1)
            # data_by_video["nonfound"].append(0)
        else:
            idx = vidids.index(vidid)
            # data_by_video["square"][idx] += 1 if square else 0
            # data_by_video["nonsquare"][idx] += 0 if square else 1
            # data_by_video["worked"][idx] += 1 if worked else 0
            # data_by_video["nonworked"][idx] += 0 if worked else 1
            data_by_video["found"][idx] += 1        

    # if square:
    #     if int(h_new) not in squarefreqs:
    #         squarefreqs[int(h_new)] = 0
    #     squarefreqs[int(h_new)] += 1

    # if vidid not in vidids:
    #     vidids.append(vidid)
    #     data_by_video["square"].append(1 if square else 0)
    #     data_by_video["nonsquare"].append(0 if square else 1)
    #     data_by_video["worked"].append(1 if worked else 0)
    #     data_by_video["nonworked"].append(0 if worked else 1)
    #     data_by_video["found"].append(1)
    #     data_by_video["nonfound"].append(0)
    # else:
    #     idx = vidids.index(vidid)
    #     data_by_video["square"][idx] += 1 if square else 0
    #     data_by_video["nonsquare"][idx] += 0 if square else 1
    #     data_by_video["worked"][idx] += 1 if worked else 0
    #     data_by_video["nonworked"][idx] += 0 if worked else 1
    #     data_by_video["found"][idx] += 1

# print("errdata")
# for i in range(errlen):
#     vidid = errdata.iloc[i]["vid"]
#     frame = errdata.iloc[i]["frame"]
#     print("{}{}".format(vidid, frame))
    
#     if vidid not in vidids:
#         vidids.append(vidid)
#         data_by_video["square"].append(0)
#         data_by_video["nonsquare"].append(0)
#         data_by_video["worked"].append(0)
#         data_by_video["nonworked"].append(0)
#         data_by_video["found"].append(0)
#         data_by_video["nonfound"].append(1)
#     else:
#         idx = vidids.index(vidid)
#         data_by_video["nonfound"][idx] += 1


pd.DataFrame(data_by_video, index=vidids).to_csv("affwild_bbox_analysis_{}.csv".format(size))

# json_obj = json.dumps(squarefreqs, indent=4)
# with open("affwild_bbox_squarefreqs.json", "w") as f:
#     f.write(json_obj)