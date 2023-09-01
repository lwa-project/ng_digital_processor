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

## Current Issues and Open Questions
 1. **SNAP2/`lwa_f`** - A few things to deal with:
     - We seem to use a different `clocksource` and OVRO-LWA -> 1 (them) to 0 (us) in `blocks/adc.py`
     - There looks to be a hard coded number of antennas -> 352\*2 (them) to 64*2 (us) in `snap2_fengine.py`
     - There looks to be some funny business with the `chan_block_id` field in the packets.  The
       lowest channel number (`chan0`) in a set appears to come out at a `chan_block_id` of `nchan_tot//nchan-2`
       rather that the zero I would expect.  I've hacked a fix into `src/formats/snap2.hpp` which shifts
       the IDs into the correct places.  _Are the data correctly oriented spectrally in the "DRX" pipelines?_
     - There is another hack in `src/formats/snap2.hpp` that relaxes the packet validation so that
       `pkt->nsrc <= _nsrc` is used instead of a `==`.
 2. **Split Networking** - I'm trying to use both network cards on the servers so I'm using a network
    splitting scheme where the SNAP2s are in 192.168.40.0/24, one NIC is on 192.168.40.0/25, and the
    the other is on 192.168.40.128/25.  I initially tried to do this with DHCP but that was a no-go.
    The configuration on the servers is now static and controlled by `netplan`.
 3. **Verbs** - Several problems getting anything to work:
     - Using OFED 5.8 for Ubuntu 20.04 with ConnectX6 cards
     - Needed verbs pacakges + the OFED kernel DMKS/tools to get things working
       (otherwise Bifrost reported "unsupported mode")
     - Works-ish now with root but permission problems for normal users (CAP_RAW_NET???)
       - I tried `spead2_net_raw` but ended up with a lot of environment variable style issues (`PYTHONPATH`,
         `LD_LIBRARY_PATH`)
     - _Do we need better core/memory/NUMA bindings to get both pipelines on a server happy?_
       - 2023/8/31: I've turned off `numactl` for the serivces for now since there are a lot of NUMA nodes
         on these servers.
 4. **`CopyOp`** - The `CopyOp` seems to stall on the resizing of `self.iring` (`capture_ring`).
    Could be a overall size problem but that seems strange:
      - LWA-SV is 500 spectra * 132 channels * 256 stands * 2 pol for a base ring size of ~32 MB
      - LWA-NA is 500 spectra * 768 channels * 64 stands * 2 pol for a base size of ~47 MB
   
    Not a huge difference but it seems to matter?
      - 2023/8/31: I currently have the buffer factor dropped by a factor of two from 10 to 5 but this still
        seems to have problems sometimes.
 5. **TBF Output** - The format needs to be updated/adpated to work with both LWA-SV and LWA-NA.  Support
    for packets containing only 64 stands is currently hacked into `src/packet_writer.hpp` and
    `src/formats/tbf.hpp` (this change is to set a 0x04 flag so that LSL knows which `fC` to use).
 6. **Correlator Sync-ing** - ~~We need a robust way to keep the correlator integrations/dumps in sync
    across pipeline restarts.~~  Switching `navg_tt` over to something that is in units of the FFT window
    length (8192 samples) **and** the block gulp size (500 * 8192 samples) seems to help.  That integration
    time works out to be 4.99461... s.
 7. **Beam Packetizer** - There seems to be a problem getting the intermediate beamformer data out of
    the "DRX" pipelines and into the T-engines.  This could be causing some back pressure that is
    interferring with the packet capture. _Is this something where verbs transmit could help?_
 8. **T-engines** - These are a large departure from what is happening with ADP at LWA-SV but similar-ish
    to what's at OVRO-LWA.  _Are all of the changes actually working?_  _Is the PFB inverter inverting
    correctly?_
 9. **"DRX" Pipelines Fight** - It's related to (8).  2023/9/1: Fixing a header problem with `ibeam1` means
    that we are back to two beams.
 10. **T-engines RX rate 0.0 B/s** - 2023/9/1: Seems to be working with two beams + fixes to the `ibeam#`
     header + new sending structure in `ndp_drx.py`.  For the header all `uint8_t` fields are now `uint16_t`.
     Packet capture looks reasonable.
 11. **`ReChannelizerOp`** - This seems related to (4) in that the resizing of the input ring causes problems.
     Both `CopyOp` and `ReChannelizerOp` are resing rings that live in `cuda_host`.  _Maybe we need to resize
     before block launch?_  _Does that even work with the capture blocks?_
 12. **kernel: watchdog: BUG: soft lockup** - These seem to have started (2023/9/1) now that the system is
     under some load.  I saw them on cetus as well but never came up with a good solution.

## The Future
 * How does the current system scale to 256 antennas?
   * How many servers do we need for a full station?
 * Is there a better clock frequency we can use?  196 MHz is good for the T-engine but lousy for the correlator
   integration time.
