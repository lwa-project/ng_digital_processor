#!/usr/bin/env python3

import timing_monitor import TimingMonitor, SYNTH_A, SYNTH_B

if __name__ == "__main__":
    import sys
    device = "/dev/ttyACM0"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    synth = TimingMonitor(device)
    #synth.set_valon_name("Sampling clock", SYNTH_A) # Note: 16 char limit
    #synth.set_valon_name("Tone injection", SYNTH_B) # Note: 16 char limit
    print("Old synth A freq:", synth.get_valon_frequency(SYNTH_A))
    synth.set_valon_freq(SYNTH_A, 196.608)
    print("New synth A freq:", synth.get_valon_freq(SYNTH_A))
    synth.save_valon_config()
