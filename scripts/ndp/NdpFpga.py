import os
import json
import argparse

try:
    from lwa_f import zcu102_fengine
    ZCU102_SUPPORT = True
except ImportError:
    ZCU102_SUPPORT = False
try:
    from lwa_f import snap2_fengine
    SNAP2_SUPPORT = True
except ImportError:
    SNAP2_SUPPORT = False
from lwa_f.error_levels import FENG_ERROR, FENG_WARNING

from filelock import FileLock

__all__ = ['SCRIPTS_PATH', 'get_lockfile', 'program', 'configure', 'check']

SCRIPTS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_lockfile(hostname):
    return f"/dev/shm/{hostname}_access.lock"
    
def _get_control_obj(hostname):
    if hostname.startswith('zcu') and ZCU102_SUPPORT:
        return zcu102_fengine.ZCU102Fengine(hostname)
    elif hostname.startswith('snap') and SNAP2_SUPPORT:
        return snap2_fengine.Snap2Fengine(hostname)
    else:
        raise RuntimeError(f"Cannot determine FPGA board type from hostname '{hostname}'")

def program(hostname, filename):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file)
    
    with access_lock:
        f = _get_control_obj(hostname)
        f.program(filename)

def configure(hostname, filename):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file)
    
    with access_lock:
        f = _get_control_obj(hostname)
        status = f.cold_start_from_config(filename)
        if not status:
            raise RuntimeError(f"Failed to configure '{hostname}'")

def check(hostname, check_style):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file)
    
    with access_lock:
        f = _get_control_obj(hostname)
        
        ret = {}
        if check_style == 'temperature':
            ret = {'error': float('nan')}
            summary, flags = f.fpga.get_status()
            ret = {'fpga': summary['temp']}
            
        elif check_style == 'operational':
            ret = {'is_ok': True,
                   'warnings': [],
                   'errors': []
                  }
            
            summary, flags = f.fpga.get_status()
            for key in flags.keys():
                if flags[key] == FENG_WARNING:
                    ret['warnings'].append(f"{key} is in warning, value is {summary[key]}")
                elif flags[key] == FENG_ERROR:
                    ret['errors'].append(f"{key} is in error, value is {summary[key]}")
                    ret['is_ok'] = False
                    
        elif check_style == 'throughput':
            ret = {'is_ok': True,
                   'gpbs': 0.0,
                   'warnings': [],
                   'errors': []
                  }
            
            summary, flags = f.eth.get_status()
            ret['gpbs'] = summary.get('gpbs', 0.0)
            if ret['gpbs'] == 0:
                ret['errors'].append("no data being sent (0 Gbps)")
                ret['is_ok'] = False
            for key in flags.keys():
                if flags[key] == FENG_WARNING:
                    ret['warnings'].append(f"{key} is in warning, value is {summary[key]}")
                elif flags[key] == FENG_ERROR:
                    ret['errors'].append(f"{key} is in error, value is {summary[key]}")
                    ret['is_ok'] = False
                    
        else:
            raise RuntimeError(f"Invalid check style mode '{check_style}'")
            
        print(json.dumps(ret))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Help for programming and configuring SNAP2 and ZCU102 FPGA boards'
    )
    parser.add_argument('operation', type=str,
                        help='operation to run, one of "program" or "configure"')
    parser.add_argument('hostname', type=str,
                        help='hostname of the FPGA board to apply the operation to')
    parser.add_argument('oparg', type=str,
                        help='filename to pass to the operation; e.g. firmware for "program", configuration file for "configure"')
    args = parser.parse_args()
    if args.operation == 'program':
        program(args.hostname, args.oparg)
    elif args.operation == 'configure':
        configure(args.hostname, args.oparg)
    elif args.operation == 'check':
        check(args.hostname, args.oparg)
    else:
        raise RuntimeError(f"Invalid operation mode '{args.operation}'")
