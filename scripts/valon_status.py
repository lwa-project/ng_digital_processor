#!/usr/bin/env python3

from timing_monitor import *

def print_synth(synth, which):
    print("Synth A" if which == SYNTH_A else "Synth B")
    print("  Freq:        ", synth.get_valon_freq(which), "MHz")
    print("  Phase locked:", synth.get_valon_lock(which))
    print("  RF enabled:  ", synth.get_valon_rf_enabled(which))

if __name__ == "__main__":
    import sys
    device = "/dev/ttyACM0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    synth = TimingMonitor(device)
    ref_names = {REF_INT: "internal",
                 REF_EXT: "external"}
    print("Ref source:", ref_names[synth.get_valon_ref_source()])
    print("Ref freq:  ", synth.get_valon_ref_freq(), "MHz")
    print_synth(synth, valon.SYNTH_A)
    print_synth(synth, valon.SYNTH_B)
