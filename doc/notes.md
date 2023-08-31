# NDP Development Notes

## Overview
 * 64 dual polarization LWA antennas
 * Two SNAP2 boards of the Caltech digitizers (32 signals/stack)
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

## Current Issues
 1. **SNAP2/`lwa_f`** - A few things to deal with:
     - We seem to use a different `clocksource` and OVRO-LWA -> 1 (them) to 0 (us) in `blocks/adc.py`
     - There looks to be a hard coded number of antennas -> 352\*2 (them) to 64*2 (us) in `snap2_fengine.py`
     - There looks to be some funny business with the `chan_block_id` field in the packets.  The
       lowest channel number (`chan0`) in a set appears to come out at a `chan_block_id` of `nchan_tot//nchan -2`
       rather that the zero I would expect.
 2. **Split Networking** - I'm trying to use both network cards on the servers so I'm using a network
    splitting scheme where the SNAP2s are in 192.168.40.0/24, one NIC is on 192.168.40.0/25, and the
    the other is on 192.168.40.128/25.  I initially tried to do this with DHCP but that was a no-go.
    The configuration on the servers is now static and controlled by `netplan`.
 3. **Verbs** - Several problems getting anything to work:
     - Using OFED 5.9 for Ubuntu 20.04 with ConnectX6 cards
     - Needed verbs pacakges + the OFED kernel DMKS/tools to get things working
       (otherwise Bifrost reported "unsupported mode")
     - Works-ish now with root but permission problems for normal users (CAP_RAW_NET???)
     - _Do we need better core bindings to get both pipelines on a server happy?_
 4. **`CopyOp`** - The `CopyOp` seems to stall on the resizing of `self.iring` (`capture_ring`).
    Could be a overall size problem but that seems strange:
      - LWA-SV is 500 spectra * 132 channels * 256 stands * 2 pol for a base ring size of ~32 MB
      - LWA-NA is 500 spectra * 768 channels * 64 stands * 2 pol for a base size of ~47 MB
   
    Not a huge difference but it seems to matter?  I currently have the buffer factor dropped by a factor
    of two from 10 to 5.
 5. **Beam packetizer** - There seems to be a problem getting the intermediate beamformer data out of
    the "DRX" pipelines and into the T-engines.  This could be causing some back pressure that is
    interferring with the packet capture. _Is this something where verbs transmit could help?_
 
