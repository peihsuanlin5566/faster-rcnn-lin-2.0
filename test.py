from pathlib import Path
import os 

DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

print(DIR_PATH.relative_to(os.path.abspath(__file__)))
# /Users/hayashi/Documents/code/20221130_fastrcnn/faster-rcnn-lin-2.0
