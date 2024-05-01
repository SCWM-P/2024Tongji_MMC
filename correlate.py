import os
import re

rootdir = os.getcwd()
savePath = os.path.join(rootdir, 'data', 'correlation')
readPath = os.path.join(rootdir, 'data', 'results')
print(rootdir, savePath, readPath)