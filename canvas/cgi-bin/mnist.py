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
sys.path.append('/Users/michaelmmeskhi/Documents/Github/TransferLRP/python/')
import lrpdemo2

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
        img_str = re.search(r'base64,(.*)', data).group(1)
        image_bytes = io.BytesIO(base64.b64decode(img_str))
        im = Image.open(image_bytes)
        im = im.resize((28,28), Image.ANTIALIAS)
        arr = np.array(im)[:,:,0:1]

        # Normalize and invert pixel values
        arr = (255 - arr) / 255.

        # Predict class
        predictions = lrpdemo2.runlrp(arr)

        # Return label data
        res['result'] = 1
        res['indices'] = sorted([int(idx) for idx in np.argpartition(predictions, -5)[-5:]])
        res['data'] = [float(num) for num in predictions[:5]] 

except Exception as e:
    # Return error data
    res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


