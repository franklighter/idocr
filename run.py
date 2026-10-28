#!/usr/bin/env python
"""便捷入口：``python run.py <图片或目录> -o results.csv``"""

import sys

from idocr.cli import main

if __name__ == "__main__":
    sys.exit(main())
