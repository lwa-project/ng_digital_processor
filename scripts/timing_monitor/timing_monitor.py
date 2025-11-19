import re
import enum
import time
import serial

from typing import Dict, Any

__all__ = ['ValonOutputs', 'ValonReferences', 'ValonSpurModes', 'TimingMonitor']


class ValonOutputs(enum.IntEnum):
    SYNTH_A = 1
    SYNTH_B = 2

class ValonReferences(enum.IntEnum):
    REF_INT = 0
    REF_EXT = 1

class ValonSpurModes(enum.IntEnum):
    LOW_NOISE_1 = 0
    LOW_NOISE_2 = 1
    LOW_SPUR_1 = 2
    LOW_SPUR_2 = 3


class TimingMonitor:
    """
    Class to interfact with a LWA Timing Monitor Box and its Valon
    synthesizer.
    """
    
    def __init__(self, port: str, baudrate: int=9600, timeout: float=1.0):
        self.port = port
        self._serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._bypass = False
        
        # Figure out if we are in bypass mode or not
        status = self._command(b'status')
        if status.find(b'Sync Pulse') == -1:
            self._bypass = True
            
        # Touch up the time on the device
        t0 = time.time()
        self._exit_bypass()
        self._command((f"TIME SET EPOCH {t0:.0f}").encode())
        
    def close(self):
        """
        Close the connection to the device.
        """
        
        self._exit_bypass()
        self._serial.close()
        
    def _command(self, cmd: bytes) -> bytes:
        """
        Low level function that takes in an encoded command, sends it to the
        timing monitor, and returns the (encoded) reply.
        """
        
        if not cmd.endswith(b'\r\n'):
            cmd += b'\r\n'
            
        self._serial.write(cmd)
        time.sleep(0.2)
        return self._serial.read_all()
        
    def _enter_bypass(self):
        """
        Put the timing monitor into bypass mode so that the user can access
        the Valon synthesizer.
        """
        
        if not self._bypass:
            resp = self._command(b'bypass on')
            if resp.startswith(b'OK'):
                resp = self._command(b'')
                self._bypass = True
            else:
                raise RuntimeError("Could not enter Valon mode")
                
    def _exit_bypass(self):
        """
        Exit the Valon sythesizer bypass mode.
        """
        
        if self._bypass:
            resp = self._command(b'+++exit')
            if resp.startswith(b'OK') or resp.startswith(b'\nOK'):
                resp = self._command(b'bypass off')
                self._bypass = False
            elif resp.startswith(b'ERR: unknown (HELP)'):
                self._bypass = False
            else:
                raise RuntimeError("Could not exit Valon mode")
                
    def get_status(self) -> Dict[str,Any]:
        """
        Query the timing monitor for its overall status and return this
        information as a dictionary.  This includes:
         * the various voltage rails it monitors
         * the Valon synthesizer's lock status
         * the sync pulse status
        """
        
        self._exit_bypass()
        resp = self._command(b'status')
        status = {}
        for line in resp.decode().split('\n'):
            if len(line) < 3:
                continue
                
            key, value = line.strip().split(':', 1)
            value = value.strip()
            try:
                value = int(value, 10)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value in ('True', 'HIGH', 'true'):
                        value = True
                    elif value == ('False', 'LOW', 'false'):
                        value = False
            status[key] = value
        return status
        
    def print_status(self):
        """
        Print out the results of get_status().
        """
        
        status = self.get_status()
        for k,v in status.items():
            print(f"{k}: {v}")
            
    def get_info(self) -> Dict[str,Any]:
        """
        Get manufacturing and build/device status information about the timing
        monitor.
        """
        
        self._exit_bypass()
        resp = self._command(b'version')
        info = {}
        for line in resp.decode().split('\n'):
            if len(line) < 3:
                continue
                
            key, value = line.strip().split(':', 1)
            value = value.strip()
            try:
                value = int(value, 10)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value in ('True', 'HIGH', 'true'):
                        value = True
                    elif value == ('False', 'LOW', 'false'):
                        value = False
            info[f"firmware_{key.lower()}"] = value
        return info
        
    def print_info(self):
        """
        Print out the results of get_info().
        """
        
        status = self.get_info()
        for k,v in status.items():
            print(f"{k}: {v}")
            
    def _valon_command(self, cmd: bytes) -> bytes:
        """
        Low level function that takes in an encoded Valon syntheizer command,
        sends it to the timing monitor via its bypass mode, and returns the
        (encoded) reply.
        """
        
        if not cmd.endswith(b'\r\n'):
            cmd += b'\r\n'
            
        self._enter_bypass()
        return self._command(cmd)
        
    def get_valon_info(self) -> Dict[str,Any]:
        """
        Get manufacturing and build/device status information about the Valon
        synthesizer.
        """
        
        resp = self._valon_command(b'id')
        resp = resp.decode().split('\r\n')[1]
        
        fields = resp.split(',')
        if len(fields) < 4:
            raise RuntimeError("Failed to determine Valon info")
            
        return {'manufacturer': fields[0].strip(),
                'model': fields[1].strip(),
                'serial_number': fields[2].strip(),
                'firmware_version': fields[3].strip().split()[1],
                'firmware_date': fields[3].strip().split('Build:')[1]
               }
        
    def print_valon_info(self):
        """
        Print out the results of get_valon_info().
        """
        
        status = self.get_valon_info()
        for k,v in status.items():
            print(f"{k}: {v}")
            
    def get_valon_ref_source(self) -> int:
        """
        Get the current Valon reference source.
        """
        
        resp = self._valon_command(b'refs?')
        resp = resp.decode()
        
        _ref = re.compile(r'REFS (?P<ref_source>\d);')
        mtch = _ref.search(resp)
        if mtch is not None:
            resp = ValonReferences(int(mtch.group('ref_source'), 10))
        else:
            raise RuntimeError("Failed to determine reference source")
        return resp
        
    def set_valon_ref_source(self, ref_source: int):
        """
        Set the Valon reference source.
        """
        
        if ref_source not in ValonReferences:
            raise ValueError(f"Invalid reference source '{source}'")
            
        self._valon_command((f"refs {ref_source}").encode())
        
    def get_valon_ref_freq(self) -> float:
        """
        Get the Valon reference source frequency (in MHz).
        """
        
        resp = self._valon_command(b'ref?')
        resp = resp.decode()
        
        _freq = re.compile(r'REF (?P<freq>\d+(\.\d*)?) MHz;')
        mtch = _freq.search(resp)
        if mtch is not None:
            resp = float(mtch.group('freq'))
        else:
            raise RuntimeError("Failed to determine refrerence frequency")
        return resp
        
    def get_valon_freq(self, source: int=ValonOutputs.SYNTH_A) -> float:
        """
        Get the Valon output frequency (in MHz) for the specified syntheizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        resp = self._valon_command(b'f?')
        resp = resp.decode()
        
        _freq = re.compile(r'F (?P<freq>\d+(\.\d*)?) MHz;')
        mtch = _freq.search(resp)
        if mtch is not None:
            resp = float(mtch.group('freq'))
        else:
            raise RuntimeError(f"Failed to determine frequency for source {source}")
        return resp
        
    def set_valon_freq(self, freq_MHz: float, source: int=ValonOutputs.SYNTH_A):
        """
        Set the Valon output frequency (in MHz) for the specified syntheizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        self._valon_command((f"F {freq_MHz}").encode())
        
    def get_valon_spur_mode(self, source: int=ValonOutputs.SYNTH_A) -> int:
        """
        Get the spur mitigation mode fro the specified syntheizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        resp = self._valon_command(b'sdn?')
        resp = resp.decode()
        
        _mod = re.compile(r'SDN (?P<mode>\d+);')
        mtch = _mod.search(resp)
        if mtch is not None:
            resp = ValonSpurModes(int(mtch.group('mode'), 10))
        else:
            raise RuntimeError(f"Failed to determine spur mitigation mode for source {source}")
        return resp
        
    def set_valon_spur_mode(self, mode: int, source: int=ValonOutputs.SYNTH_A):
        """
        Set the spur mitigation mode fro the specified syntheizer.
        """
        
        if mode not in ValonSpurModes:
            raise ValueError(f"Invalid spur mitigation mode '{mode}'")
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        self._valon_command((f"sdn {mode}").encode())
        
    def get_valon_atten(self, source: int=ValonOutputs.SYNTH_A) -> float:
        """
        Get the internal attenuator setting (in dB) for the specified syntheizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        resp = self._valon_command(b'att?')
        resp = resp.decode()
        
        _atten =  re.compile(r'ATT (?P<atten>\d+(\.\d*)?);')
        mtch = _atten.search(resp)
        if mtch is not None:
            resp = float(mtch.group('atten'))
        else:
            raise RuntimeError(f"Failed to determine attenuator setting for source {source}")
        return resp
        
    def set_valon_atten(self, value_dB,  source: int=ValonOutputs.SYNTH_A):
        """
        Set the internal attenuator setting (in dB) for the specified syntheizer.
        """
        
        value_dB = int(round(value_dB*4))/4.0
        if value_dB < 0 or value_dB > 31.75:
            raise ValueError(f"Invalid attenuation setting '{value_dB}'")
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        self._valon_command((f"att {value_dB:.2f}").encode())
        
    def get_valon_rf_enabled(self, source: int=ValonOutputs.SYNTH_A) -> bool:
        """
        Get the RF output state for the specified synthesizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        resp = self._valon_command(b'oen?')
        resp = resp.decode()
        
        _on = re.compile(r'OEN (?P<enabled>\d);')
        mtch = _on.search(resp)
        if mtch is not None:
            resp = (mtch.group('enabled') == '1')
        else:
            raise RuntimeError(f"Failed to determine RF output state for source {source}")
        return resp
        
    def set_valon_rf_enabled(self, turn_on: bool, source: int=ValonOutputs.SYNTH_A):
        """
        Get the RF output state for the specified synthesizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        value = 1 if turn_on else 0
        self._valon_command((f"source {source}").encode())
        self._valon_command((f'oen {value}').encode())
        
    def get_valon_name(self, source: int=ValonOutputs.SYNTH_A) -> str:
        """
        Get the name of the specified synthesizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        resp = self._valon_command(b'name?')
        resp = resp.decode()
        
        _name = re.compile(r'NAME (?P<name>.*?);')
        mtch = _name.search(resp)
        if mtch is not None:
            resp = mtch.group('name')
        else:
            raise RuntimeError(f"Failed to determine name for source {source}")
        return resp
        
    def set_valon_name(self, name, source: int=ValonOutputs.SYNTH_A):
        """
        Set the name of the specified synthesizer.
        """
        
        if len(name) > 16:
            print("WARNING: truncating name to 16 characters")
            name = name[:16]
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        self._valon_command((f"source {source}").encode())
        self._valon_command((f"name {name}").encode())
        
    def get_valon_lock(self, source: int=ValonOutputs.SYNTH_A) -> bool:
        """
        Get the Valon output lock status for the specified syntheizer.
        """
        
        if source not in ValonOutputs:
            raise ValueError(f"Invalid source '{source}'")
            
        resp = self._valon_command((f"lock{source}?").encode())
        resp = resp.decode()
        
        if resp.find(f"S{source} locked") != -1:
            return True
        else:
            return False
            
    def save_valon_config(self):
        """
        Write the Valon configuration to flash.
        """
        
        self._valon_command(b'save')
        
