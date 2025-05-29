#!/usr/bin/env python3

# Simple script to interact with NDP via MCS commands without having to
# have the full station MCS running.

import sys
import time
from queue import Queue
import struct

from ndp import MCS2

MSG_REPLY_TIMEOUT = 5


if __name__ == '__main__':
    sender = MCS2.MsgSender(("localhost",1742), subsystem='MCS')
    sender.input_queue = Queue()
    receiver = MCS2.MsgReceiver(("0.0.0.0",1743))
    sender.daemon = True
    receiver.daemon = True
    sender.start()
    receiver.start()
    
    cmd = sys.argv[1]
    data = ' '.join(sys.argv[2:])
    
    # Convert command data to binary as needed
    if cmd == 'DRX':
        beam, tune, freq, bw, gain, high_dr, subslot = data.split(None, 6)
        beam = int(beam, 10)
        tune = int(tune, 10)
        freq = float(freq)
        bw = int(bw, 10)
        gain = int(gain, 10)
        high_dr = int(high_dr, 10)
        subslot = int(subslot, 10)
        data = struct.pack('>BfBh', beam, tune, freq, bw, gain, high_dr, subslot)
        
    msg = MCS2.Msg(dst='NDP', cmd=cmd, data=data)
    print(f"Sending {msg}")
    sender.put(msg)
    reply = receiver.get(timeout=MSG_REPLY_TIMEOUT)
    if reply is not None:
        print(f"Recieved {reply}")
        if len(reply.data)-8 == 4:
            ## Could be a float...
            print(struct.unpack('>f', reply.data[8:]))
            
    else:
        print("No reply received")
        
    sender.request_stop()
    receiver.request_stop()
    sender.join()
    receiver.join()
