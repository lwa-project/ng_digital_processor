from .NdpCommon import CHAN_BW

from lwa_f import snap2_fengine 

import os
import yaml
import iptools

def connect_to_snap(hostname):
    return snap2_fengine.Snap2Fengine(hostname)] 


def write_snap_config(config, filename):
    yconf = {'fengines': {'enable_pfb': not config['snap']['bypass_pfb'],
                          'fft_shift': config['snap']['fft_shift'],
                          'chans_per_packet': 96},
             'xengines:' {'arp': {}, 'chans': {}}
             
    try:
        equalizer_coeffs = np.loadtxt(config['snap']['equalizer_coeffs'])
        yconfig['fengines']['eq_coeffs'] = list(equalizer_coeffs)
    except Exception as e:
        pass
        
    for i,snap in enumerate(config['host']['snaps']):
        yconf['fengines'][snap]['ants'] = [i*32, (i+1)*32]
        yconf['fengines'][snap]['gbe'] = iptools.int2ip(iptools.ip2int(config['snap']['data_ip_base']) + i)
        yconf['fengines'][snap]['source_port'] = config['snap']['data_port_base']
        
    macs = iptools.load_ethers()
    yconfig['xengines']['arp'].update(macs)
    
    for i,ip in enumerate(macs.keys()):
        for j in range(2):
            chan0 = config['drx'][2*i + j]['first_channel']
            nchan = int(round(config['drx'][2*i + j]['capture_bandwidth'] / CHAN_BW))
            port = 10000*(i+1)
        
            yconfig['xengines']['chans'][f"{ip}-{port}"] = [chan0, chan0+nchan]
            
    yaml.dump(yconfig, filename)
    return os.path.getsize(filename)
