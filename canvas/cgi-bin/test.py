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

nnt, nnb = lrpdemo2.loadmodels()

print(nnb)
