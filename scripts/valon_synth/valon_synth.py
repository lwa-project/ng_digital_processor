# Copyright (C) 2011 Associated Universities, Inc. Washington DC, USA.
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
# 
# Correspondence concerning GBT software should be addressed as follows:
#	GBT Operations
#	National Radio Astronomy Observatory
#	P. O. Box 2
#	Green Bank, WV 24944-0002 USA

"""
Provides a serial interface to the Valon 500x.
"""

from __future__ import division

# Python modules
import struct
# Third party modules
import serial


__author__ = "Patrick Brandt"
__copyright__ = "Copyright 2011, Associated Universities, Inc."
__credits__ = ["Patrick Brandt, Stewart Rumley, Steven Stark"]
__license__ = "GPL"
#__version__ = "1.0"
__maintainer__ = "Patrick Brandt"


# Handy aliases
SYNTH_A = 0x01
SYNTH_B = 0x02

INT_REF = 0x00
EXT_REF = 0x01

ACK = 0x06
NACK = 0x15

class Synthesizer:
    def __init__(self, port):
        self.conn = serial.Serial(port, 9600, serial.EIGHTBITS,
                                  serial.PARITY_NONE, serial.STOPBITS_ONE,
                                  timeout=1)
        self.conn.close()

    def _send_command(self, command):
        try:
            command = command.encode()
        except AttributeError:
            # Python2 catch
            pass
            
        self.conn.open()
        self.conn.write(command+b'\r\n')
        response = self.conn.readline()
        response = self.conn.readline()
        self.conn.close()
        
        try:
            response = response.decode()
        except AttributeError:
            # Python2 catch
            pass
        return response

    @staticmethod
    def _parse_simple(value):
        try:
            value, _ = value.split(';', 1)
        except ValueError:
            pass
        value = value.rstrip()
        try:
            _, value = value.rsplit(None, 1)
        except ValueError:
            pass
        try:
            value = int(value, 10)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        return value

    @staticmethod
    def _parse_frequency(value):
        value, _ = value.split(';', 1)
        value = value.rstrip()
        _, value, unit = value.rsplit(None, 2)
        value = float(value)
        if unit == 'MHz':
            value *= 1e6
        elif unit == 'kHz':
            value *= 1e3
        return value

    def get_frequency(self, synth):
        """
        Returns the current output frequency for the selected synthesizer.

        @param synth : synthesizer this command affects (0 for 1, 8 for 2).
        @type  synth : int

        @return: the frequency in MHz (float)
        """
        freq = self._send_command('F?')
        return self._parse_frequency(freq)/1e6

    def set_frequency(self, synth, freq, chan_spacing = 10.):
        """
        Sets the synthesizer to the desired frequency

        Sets to the closest possible frequency, depending on the channel spacing.
        Range is determined by the minimum and maximum VCO frequency.

        @param synth : synthesizer this command affects.
        @type  synth : int

        @param freq : output frequency
        @type  freq : float

        @param chan_spacing : output frequency increment
        @type  chan_spacing : float

        @return: True if success (bool)
        """
        return self._send_command('S %i; F %.3f' % (synth, freq))

    def get_reference(self):
        """
        Get reference frequency in MHz
        """
        freq = self._send_command('REF?')
        return self._parse_frequency(freq)

    def set_reference(self, freq):
        """
        Set reference frequency in MHz

        @param freq : frequency in MHz
        @type  freq : float

        @return: True if success (bool)
        """
        return self._send_command('REF %.3f' % freq)

    def get_rf_level(self, synth):
        """
        Returns RF level in dBm

        @param synth : synthesizer, 1 or 2
        @type  synth : int

        @return: dBm (int)
        """
        rfl_table = {0: -4, 1: -1, 2: 2, 3: 5}
        rfl = self._send_command('S %i; PLEV?' % synth)
        return rfl_table[self._parse_simple(rfl)]

    def set_rf_level(self, synth, rf_level):
        """
        Set RF level

        @param synth : synthesizer, 1 or 2
        @type  synth : int

        @param rf_level : RF power in dBm
        @type  rf_level : int

        @return: True if success (bool)
        """
        rfl_rev_table = {-4: 0, -1: 1, 2: 2, 5: 3}
        rfl = rfl_rev_table.get(rf_level)
        if(rfl is None): return False
        return self._send_command('S %i; PLEV %i' % (synth, rfl))

    def get_rf_output_enabled(self, synth):
        onoff = self._send_command('S %i; OEN?' % synth)
        return bool(self._parse_simple(onoff))

    def set_rf_output_enabled(self, synth, enabled):
        return self._send_command('S %i; OEN %i' % (synth, enabled))
        
    def get_options(self, synth):
        """
        Get options tuple:

        bool double:   if True, reference frequency is doubled
        bool half:     if True, reference frequency is halved
        int  r:        reference frequency divisor
        bool low_spur: if True, minimizes PLL spurs;
                       if False, minimizes phase noise
        double and half both True is same as both False.

        @param synth : synthesizer address

        @return: double (bool), half (bool), r (int), low_spur (bool)
        """
        double = self._send_command('S %i; REFDB?' % synth)
        double = bool(self._parse_simple(double))
        half = self._send_command('S %i; REFDIV?' % synth)
        half = bool(self._parse_simple(half))
        r = 1
        low_spur = self._send_command('S %i; SDN?' % synth)
        low_spur = self._parse_simple(low_spur) == 0
        return double, half, r, low_spur

    def set_options(self, synth, double = 0, half = 0, r = 1, low_spur = 0):
        """
        Set options.
        
        double and half both True is same as both False.

        @param synth : synthesizer base address
        @type  synth : int
        
        @param double : if 1, reference frequency is doubled; default 0
        @type  double : int
        
        @param half : if 1, reference frequency is halved; default 0
        @type  half : int
        
        @param r : reference frequency divisor; default 1
        @type  r : int
        
        @param low_spur : if 1, minimizes PLL spurs;
                          if 0, minimizes phase noise; default 0
        @type  low_spur : int

        @return: True if success (bool)
        """
        self._send_command('S %i; REFDB %i' % (synth, double))
        self._send_command('S %i; REFDIV %i' % (synth, half))
        return self._send_command('S %i; SDN %i' % (synth, 11*low_spur))

    def get_ref_select(self):
        """Returns the currently selected reference clock.

        Returns 1 if the external reference is selected, 0 otherwise.
        """
        response = self._send_command('REFS?')
        return self._parse_simple(response)

    def set_ref_select(self, e_not_i = 1):
        """
        Selects either internal or external reference clock.

        @param e_not_i : 1 (external) or 0 (internal); default 1
        @type  e_not_i : int

        @return: True if success (bool)
        """
        return self._send_command('REFS %i' % e_not_i)
        
    def get_vco_range(self, synth):
        """
        Returns (min, max) VCO range tuple.

        @param synth : synthesizer base address
        @type  synth : int

        @return: min,max in MHz
        """
        self.conn.open()
        bytes = struct.pack('>B', 0x83 | synth)
        self.conn.write(bytes+b'\r\n')
        bytes = self.conn.read(4)
        checksum = self.conn.read(1)
        self.conn.close()
        #self._verify_checksum(bytes, checksum)
        min, max = struct.unpack('>HH', bytes)
        return min, max

    def set_vco_range(self, synth, min, max):
        """
        Sets VCO range.

        @param synth : synthesizer base address
        @type  synth : int

        @param min : minimum VCO frequency
        @type  min : int

        @param max : maximum VCO frequency
        @type  max : int

        @return: True if success (bool)
        """
        self.conn.open()
        bytes = struct.pack('>BHH', 0x03 | synth, min, max)
        checksum = self._generate_checksum(bytes)
        self.conn.write(bytes + checksum + b'\r\n')
        bytes = self.conn.read(1)
        self.conn.close()
        ack = struct.unpack('>B', bytes)[0]
        return ack == ACK

    def get_phase_lock(self, synth):
        """
        Get phase lock status

        @param synth : synthesizer base address
        @type  synth : int

        @return: True if locked (bool)
        """
        ref = self.get_ref_select()+1
        locked = self._send_command('LK%i?' % ref)
        locked = self._parse_simple(locked)
        return locked == 'locked'

    def get_label(self, synth):
        """
        Get synthesizer label or name

        @param synth : synthesizer base address
        @type  synth : int

        @return: str
        """
        label = self._send_command('S %i; NAM?' % synth)
        return self._parse_simple(label)

    def set_label(self, synth, label):
        """
        Set synthesizer label or name

        @param synth : synthesizer base address
        @type  synth : int

        @param label : up to 16 bytes of text
        @type  label : str
        
        @return: True if success (bool)
        """
        return self._send_command('S %i, NAM %s' % (synth, label))

    def flash(self):
        """
        Flash current settings for both synthesizers into non-volatile memory.

        @return: True if success (bool)
        """
        return self._send_command('SAV')

    def _getEPDF(self, synth):
        """
        Returns effective phase detector frequency.

        This is the reference frequency with options applied.
        """
        reference = self.get_reference() / 1e6
        double, half, r, low_spur = self.get_options(synth)
        if(double): reference *= 2.0
        if(half):   reference /= 2.0
        if(r > 1):  reference /= r
        return reference
