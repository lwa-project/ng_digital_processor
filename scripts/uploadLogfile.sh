#!/bin/bash

ls /home/ndp/log/*.gz | xargs -n1 /home/ndp/ng_digital_processor/scripts/uploadLogfile.py
