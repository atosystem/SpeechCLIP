import os
import glob
import json

images_path = "./Images/*.jpg"
output_pairID_json = "./Flickr8k_idPairs.json"


splitnames = ["train","dev","test"]
filenamesBySplit = { _split:[] for _split in splitnames}

for _splitname in splitnames:
    txt_fp = "./Flickr_8k.{}Images.txt".format(_splitname)
    with open(txt_fp,"r") as f:
        _data = f.readlines()
    for _fn in _data:
        _fn = _fn.strip()
        if _fn == "": continue
        filenamesBySplit[_splitname].append(
            os.path.basename(_fn).replace(".jpg","")
        )

all_img_list = sorted(glob.glob(images_path))

print("Total Images found #{}".format(len(all_img_list)))

all_img_list = [ os.path.basename(x).replace(".jpg","") for x in all_img_list]

id2Filename = all_img_list

filename2Id = { x:i for i,x in enumerate(id2Filename)}


filenamesWithSplitName = []

for x in id2Filename:
    _x_split = None
    for _splitname in splitnames:
        if x in filenamesBySplit[_splitname]:
            if _x_split is not None:
                print("Duplicate filenames for {}, found in {} and {}".format(x,_x_split,_splitname))
                exit(1)
            else:
                _x_split = _splitname
    filenamesWithSplitName.append(
        (
            x, _x_split
        )
    )

for _splitname in splitnames:
    print("\t{} #{}".format(_splitname,len(filenamesBySplit[_splitname])))


    
with open(output_pairID_json,"w") as f:
    json.dump({
        "id2Filename" : id2Filename,
        "filename2Id" : filename2Id,
        "filenamesWithSplitName" : filenamesWithSplitName,
    },f)

print("Saved ids to {}".format(output_pairID_json))




