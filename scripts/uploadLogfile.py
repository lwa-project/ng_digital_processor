#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from socket import gethostname

from lwa_auth import KEYS as LWA_AUTH_KEYS
from lwa_auth.signed_requests import post as signed_post

URL = "https://lda10g.alliance.unm.edu/metadata/sorter/upload"
SITE = "lwana"
TYPE = "SSLOG"

# Send the update to lwalab
r = os.path.realpath(sys.argv[1])
f = signed_post(LWA_AUTH_KEYS.get('ndp', kind='private'), URL,
                data={'site': SITE, 'type': TYPE, 'subsystem': 'NDP'},
                files={'file': open(r, 'rb'))
print(f.text)
f.close()
