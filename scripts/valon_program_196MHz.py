#!/usr/bin/env python3

import os
import sys
tm_path = os.path.dirname(os.path.abspath(__file__))
if tm_path not in sys.path:
    sys.path.insert(0, tm_path)

from timing_monitor import *

if __name__ == "__main__":
    import sys
    device = "/dev/ttyACM0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    synth = TimingMonitor(device)
    #synth.set_valon_name("Sampling clock", ValonOutputs.SYNTH_A) # Note: 16 char limit
    #synth.set_valon_name("Tone injection", ValonOutputs.SYNTH_B) # Note: 16 char limit
    print("Old synth A freq:", synth.get_valon_freq(ValonOutputs.SYNTH_A))
    synth.set_valon_freq(ValonOutputs.SYNTH_A, 196.0)
    print("New synth A freq:", synth.get_valon_freq(ValonOutputs.SYNTH_A))
    synth.save_valon_config()
