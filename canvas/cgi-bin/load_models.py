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
models = {}

class loadnn(object):
    def __init__(self):
        self.target, self.base = lrpdemo2.loadmodels()

try:
    # Load models
    if os.environ["REQUEST_METHOD"] == "GET":
    	nn = loadnn()

except Exception as e:
    # Return error data
    models['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(nn, default=lambda x: x.__dict__))


