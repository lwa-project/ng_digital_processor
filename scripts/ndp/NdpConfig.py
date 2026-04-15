
import os
try:
    import simplejson as json
except ImportError:
    print("Warning: Failed to import simplejson; falling back to vanilla json")
    import json

def parse_config_file(filename, log=None):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config
