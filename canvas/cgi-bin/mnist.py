#!/usr/bin/env python3

"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""

import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
sys.path.append('/Users/michaelmmeskhi/Desktop/TEX/python/')
import lrpfinal
import math

# Default output
res = {"result": 0,
       "data": [], 
       "error": '',
       "indices": []}

try:
    # Get post data
    if os.environ["REQUEST_METHOD"] == "POST":
        data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))
        # Convert data url to numpy array
        # img_str = re.search(r'base64,(.*)', data).group(1)
        img_idx = data.index(',')
        var_idx = data.index('?')
        img_str = data[img_idx+1:var_idx]
        var = data[var_idx+1:]

        model = var[var.index('mdl')+3:var.index('lrp')]
        lrp = var[var.index('lrp')+3:var.index('mtd')]
        method = int(var[var.index('mtd')+3:var.index('mth')])
        methodT = int(var[var.index('mth')+3:var.index('oth')])
        overlapT = int(var[var.index('oth')+3:])

        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        im = im.resize((28,28), Image.ANTIALIAS)
        im.save("/Users/michaelmmeskhi/Desktop/TEX/canvas/images/input.png")
        arr = np.array(im)[:,:,0:1]

        # Normalize and invert pixel values
        arr = (255 - arr)

        # Predict class
        predictions = lrpfinal.run_demo(arr, model, lrp, method, methodT, overlapT)
        top = [float(num) for num in predictions]
        pred = max(top)
        # ind = []
        # predictions = list(predictions)
        # for x in top:
        #   ind.append(predictions.index(x))
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        top = list(top)
        secondMax = -math.inf
        for x in top:
            if x<pred and x > secondMax:
                secondMax = x
        ind = top.index(pred)
        ind2 = top.index(secondMax)

        if len(predictions) == 26:
            ind = letters[ind]
            ind2 = letters[ind2]

        # Return label data
        res['result'] = 1
        res['indices'] = [ind, ind2]
        res['data'] = top

except Exception as e:
    # Return error data
    res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


