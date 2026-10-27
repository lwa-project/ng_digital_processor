import os
import sys
import json
import time
import numpy as np
import argparse
import contextlib

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

from .NdpCommon import CHAN_BW

__all__ = ['SCRIPTS_PATH', 'LOCK_TIMEOUT', 'get_lockfile', 'program', 'configure', 'spectra', 'check']

SCRIPTS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_TIMEOUT = 30  # seconds

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
    access_lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)
    
    with access_lock:
        f = _get_control_obj(hostname)
        f.program(filename)

def configure(hostname, filename):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)
    
    with access_lock:
        f = _get_control_obj(hostname)
        status = f.cold_start_from_config(filename)
        if not status:
            raise RuntimeError(f"Failed to configure '{hostname}'")

def spectra(hostname, t_int):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)
    
    acc_len = int(round(CHAN_BW * t_int))
    with access_lock:
        f = _get_control_obj(hostname)
        
        spectra = []
        f.autocorr.set_acc_len(acc_len)
        time.sleep(t_int + 0.1)
        
        for i in range(f.autocorr._n_cores):
            spectra.append(f.autocorr.get_new_spectra(signal_block=i))
            
    spectra = np.array(spectra)
    spectra = spectra.reshape(-1, spectra.shape[-1])
    
    filename = f"/dev/shm/{hostname}_spectra_{os.getpid()}.npy"
    np.save(filename, spectra)
    return {'filename': filename,
            'shape': list(spectra.shape),
            'dtype': str(spectra.dtype)}

def equalizer(hostname):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)
    
    ninput = 64
    if hostname.startswith('zcu'):
        ninput = 32
        
    with access_lock:
        f = _get_control_obj(hostname)
        
        old_coeffs = []
        for i in range(ninput):
            old_coeffs.append(f.eq.get_coeffs(i))
            
        f.set_equalization()
        
        coeffs = []
        for i in range(ninput):
            coeffs.append(f.eq.get_coeffs(i))
            f.eq.set_coeffs(i, old_coeffs[i])
            
    coeffs = np.array(coeffs)
    
    filename = f"/dev/shm/{hostname}_coeffs_{os.getpid()}.npy"
    np.save(filename, coeffs)
    return {'filename': filename,
            'shape': list(coeffs.shape),
            'dtype': str(coeffs.dtype)}

def _histogram_coverage(vals, hist):
    # Find the fraction of ADC values within +/-1 sigma of the median that
    # are actually occupied.  The standard deviation comes from the 16th/84th
    # percentiles so that it is not dominated by outliers.  A real analog
    # signal, low or high power, fills every value near the median.  Misaligned
    # ADC bits leave holes (only every 2nd/4th/64th value is used).
    #
    # Returns a two-element tuple of (sigma, coverage).
    
    cdf = np.cumsum(hist) / hist.sum()
    med = vals[np.searchsorted(cdf, 0.50)]
    sig = (vals[np.searchsorted(cdf, 0.84)] - vals[np.searchsorted(cdf, 0.16)]) / 2
    sig = max(sig, 1)
    in_core = (vals >= med - sig) & (vals <= med + sig)
    coverage = np.count_nonzero(hist[in_core] > 0) / np.count_nonzero(in_core)
    return sig, coverage

def _histogram_is_dead(hist):
    # Is this narrow histogram a dead (no analog signal) input or is it Swiss
    # cheese?  On a healthy system dead inputs fill a single solid block of 4
    # to 8 ADC values, possibly offset from zero by a few counts.  Broken
    # inputs that happen to be narrow instead have holes (e.g., only every
    # other value) or stray values far away from the median.  Values with less
    # than 1e-4 of the samples are ignored so that a stray outlier does not
    # matter.
    #
    # Returns True if the histogram is a solid block of <= 12 values with at
    # most one empty value inside it.
    
    used = np.where(hist / hist.sum() >= 1e-4)[0]
    span = used[-1] - used[0] + 1
    return (span <= 12) and (span - len(used) <= 1)

def check(hostname, check_style):
    lock_file = get_lockfile(hostname)
    access_lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)
    
    with access_lock:
        f = _get_control_obj(hostname)
        
        ret = {}
        if check_style == 'temperature':
            ret = {'error': float('nan')}
            summary, flags = f.fpga.get_status()
            ret = {'fpga': summary['temp']}
            
        elif check_style == 'rms':
            ret = {'min': 0.0,
                   'mean': 0.0,
                   'max': 0.0,
                   'std': 0.0
                  }
            
            _, _, rmss = f.input.get_bit_stats()
            ret['min'] = rmss.min()
            ret['mean'] = rmss.mean()
            ret['max'] = rmss.max()
            ret['std'] = rmss.std()

        elif check_style == 'histogram':
            # This needs some explination:  look for ADC inputs whose histograms
            # have holes, which is the signature of lost ADC lane alignment
            # (sometimes seems after power events that aren't long enough to
            # take down the servers).  The standard deviation is estimated from
            # the wings of the histogram (16th/84th percentile) and then:
            #  * Wide (sigma >= 3 counts):  A live signal.  Every ADC value
            #    within +/-1 sigma of the median should be seen.  If not, the
            #    input is bad.
            #  * Narrow (sigma < 3 counts):  Probably a dead input.  If it is
            #    a solid block of values it is 'unknown' (an analog problem we
            #    cannot judge here).  If it is Swiss cheese it is bad.
            ret = {'nstream': 0,
                   'nbad': 0,
                   'nunknown': 0,
                   'frac_bad': 0.0,
                   'bad': [],
                   'unknown': []
                  }
            
            for i in range(f.input.n_streams):
                vals, hist_a, hist_b = f.input.get_histogram(i, sum_cores=False)
                vals = np.array(vals)
                hist_a = np.array(hist_a, dtype=float)
                hist_b = np.array(hist_b, dtype=float)
                sig_a, cov_a = _histogram_coverage(vals, hist_a)
                sig_b, cov_b = _histogram_coverage(vals, hist_b)
                
                if min(sig_a, sig_b) >= 3:
                    if min(cov_a, cov_b) < 0.9:
                        ret['bad'].append(i)
                elif _histogram_is_dead(hist_a + hist_b):
                    ret['unknown'].append(i)
                else:
                    ret['bad'].append(i)
            
            ret['nstream'] = f.input.n_streams
            ret['nbad'] = len(ret['bad'])
            ret['nunknown'] = len(ret['unknown'])
            ret['frac_bad'] = ret['nbad'] / ret['nstream']
            
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
                   'gbps': 0.0,
                   'warnings': [],
                   'errors': []
                  }
            
            summary, flags = f.eth.get_status()
            ret['gbps'] = summary.get('gbps', 0.0)
            if ret['gbps'] == 0:
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
            
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Help for programming and configuring SNAP2 and ZCU102 FPGA boards'
    )
    parser.add_argument('operation', type=str,
                        help='operation to run, one of "program", "configure", "spectra", "equalizer", or "check"')
    parser.add_argument('hostname', type=str,
                        help='hostname of the FPGA board to apply the operation to')
    parser.add_argument('oparg', type=str,
                        help='filename to pass to the operation; e.g. firmware for "program", configuration file for "configure"')
    args = parser.parse_args()
    
    ret = None
    with contextlib.redirect_stdout(sys.stderr):
        if args.operation == 'program':
            program(args.hostname, args.oparg)
        elif args.operation == 'configure':
            configure(args.hostname, args.oparg)
        elif args.operation == 'spectra':
            ret = spectra(args.hostname, float(args.oparg))
        elif args.operation == 'equalizer':
            ret = equalizer(args.hostname)
        elif args.operation == 'check':
            ret = check(args.hostname, args.oparg)
        else:
            raise RuntimeError(f"Invalid operation mode '{args.operation}'")
            
    if ret is not None:
        print(json.dumps(ret))
