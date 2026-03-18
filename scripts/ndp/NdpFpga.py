import os
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
from lwa_f.error_levels import FENG_ERROR as FENG_ERROR_CODE
from filelock import FileLock

__all__ = ['SCRIPTS_PATH', 'get_lockfile', 'program', 'configure']

SCRIPTS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_lockfile(hostname):
    return f"/dev/shm/{hostname}_access.lock"

def program(hostname, filename):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file)
    
    with access_lock:
        if hostname.startswith('zcu') and ZCU102_SUPPORT:
            f = zcu102_fengine.ZCU102Fengine(hostname)
        elif hostname.startswith('snap') and SNAP2_SUPPORT:
            f = snap2_fengine.Snap2Fengine(hostname)
        else:
            raise RuntimeError(f"Cannot determine FPGA board type from hostname '{hostname}'")
            
        f.program(filename)

def configure(hostname, filename):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file)
    
    with access_lock:
        if hostname.startswith('zcu') and ZCU102_SUPPORT:
            f = zcu102_fengine.ZCU102Fengine(hostname)
        elif hostname.startswith('snap') and SNAP2_SUPPORT:
            f = snap2_fengine.Snap2Fengine(hostname)
        else:
            raise RuntimeError(f"Cannot determine FPGA board type from hostname '{hostname}'")
            
        status = f.cold_start_from_config(filename)
        if not status:
            raise RuntimeError(f"Failed to configure '{hostname}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Help for programming and configuring SNAP2 and ZCU102 FPGA boards'
    )
    parser.add_argument('operation', type=str,
                        help='operation to run, one of "program" or "configure"')
    parser.add_argument('hostname', type=str,
                        help='hostname of the FPGA board to apply the operation to')
    parser.add_argument('oparg', type=str,
                        help='filename to pass to the operation; firmware for "program", configuration file for "configure"')
    args = parser.parse_args()
    if args.operation == 'program':
        program(args.hostname, args.oparg)
    elif args.operation == 'configure':
        configure(args.hostname, args.oparg)
    else:
        raise RuntimeError(f"Invalid operation mode '{args.operations}'")
