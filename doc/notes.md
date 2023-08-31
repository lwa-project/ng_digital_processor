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

 
