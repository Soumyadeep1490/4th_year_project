import os
import sys

# current directory
_dir = os.getcwd()

# bring the parent directory into the scope
sys.path.append(os.path.join(_dir, '..'))
