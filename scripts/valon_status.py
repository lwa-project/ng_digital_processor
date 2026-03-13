#!/usr/bin/env python3

import os
import sys
tm_path = os.path.dirname(os.path.abspath(__file__))
if tm_path not in sys.path:
    sys.path.insert(0, tm_path)
    
from timing_monitor import *

def print_synth(synth, which):
    print(which.name)
    print("  Freq:        ", synth.get_valon_freq(which), "MHz")
    print("  Phase locked:", synth.get_valon_lock(which))
    print("  RF enabled:  ", synth.get_valon_rf_enabled(which))
    try:
        sdn = synth.get_valon_spur_mode(which)
        print("  Spur mode:   ", sdn.name)
    except RuntimeError:
        print("  Spur mode:   ", "unknown")

if __name__ == "__main__":
    import sys
    device = "/dev/ttyACM0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    synth = TimingMonitor(device)
    synth.print_status()
    info = synth.get_valon_info()
    print("Model:", info['model'])
    ref = synth.get_valon_ref_source()
    print("Ref source:", ref.name)
    print("Ref freq:  ", synth.get_valon_ref_freq(), "MHz")
    print_synth(synth, ValonOutputs.SYNTH_A)
    print_synth(synth, ValonOutputs.SYNTH_B)
