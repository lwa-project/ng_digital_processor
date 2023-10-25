# NDP Development Notes

## Overview
 * 64 dual polarization LWA antennas
 * Two SNAP2 boards with the Caltech digitizers
   - 32 signals/dig. stack, 64 signals/SNAP2
   - Sample rate is 196 MHz
 * Three GPU servers - two with A5000, one with A4000
 * Sending 3072 channels in sets of 768 to four "DRX" pipelines
   - Packets are eight sets of 96 channels from each SNAP2 board
   - Data are 4+4-bit complex integer
   - Data rate per pipeline is ~2.2 GB/s
 * "DRX" pipelines are:
   - Triggered dump (TBF) - currently going to disk
   - Initial beamformer (four dual pol. beams)
   - Correlator
 * After "DRX" are:
   - beamformer -> T-engines
   - correlator -> Orville-style imager

## Implementation Notes
 1. **Split Networking** - I'm trying to use both network cards on the servers so I'm using a network
    splitting scheme where the SNAP2s are in 192.168.40.0/24, one NIC is on 192.168.40.0/25, and the
    the other is on 192.168.40.128/25.  I initially tried to do this with DHCP but that was a no-go.
    The configuration on the servers is now static and controlled by `netplan`.
 2. **`bifrost.map` Cache** - There was a problem with having the `bifrost.map` cache stored in `~/.bifrost`
    because of NFS file locking problems.  There wasn't always a deadlock but it happened often enough
    that I moved the map cache to `/opt/.bifrost` inside the Bifrost code.
 3. **Packet Format Changes** - I have made several modifications to the intermediate and output data
    products to get things working:
     * The TBF `unassigned` header field is now used to store the number of stands in a TBF frame.
       0 is assumed to be 256 for backward compatibility with LWA-SV/ADP.
     * The TBF ID has been updated to 0x05; 0x01 for TBF or'd with 0x04 for NDP.
     * The COR ID has been updated to 0x06; 0x02 for COR or'd with 0x04 for NDP.
     * The COR `navg` header field is now in units of ticks of a 196 MHz clock.
 4. **GPU Swap** - The A4000s that were in ndp0 have been moved to ndp2.  Now ndp0 and ndp1 have
    the A5000s.  This helps reduce the GPU occupancy on the T-engine.
 5. **ndp2 Repurposed** - It looks like I can squeeze the bandwidth down into two DRX pipelines,
    freeing up ndp2 to be an imager.  We need a plan for what to do with all of the Orville data.
    
## Problems and Open Questions
See https://github.com/lwa-project/ng_digital_processor/issues

## The Future
 * How does the current system scale to 256 antennas?
   * How many servers do we need for a full station?
 * Is there a better clock frequency we can use?  196 MHz is good for the T-engine but lousy for the correlator
   integration time.
