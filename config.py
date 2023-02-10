import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# WEIGHTS = ROOT / 'runs/train/exp2/weights/best.pt'
WEIGHTS = ROOT / 'best.pt'
IMGSZ = [640, 640]
CONF_THRES = 0.3
IOU_THRES = 0.45
MAX_DET = 1000
LINE_THICKNESS = 1  # 线框粗细

# HIDE_CONF = False  # 隐藏置信度
# HIDE_LABELS = False  # 隐藏标签
