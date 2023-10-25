
"""
Inter-Server Communication (ISC) for NDP.  This allows messages to be send to 
the different servers for on-the-fly pipeline reconfiguration.
"""

import zmq
import time
import numpy
import binascii
import threading
from uuid import uuid4
from collections import deque


__version__ = '0.6'
__all__ = ['logException', 'PipelineMessageServer', 'StartTimeClient', 'TriggerClient',
           'DRXConfigurationClient', 'BAMConfigurationClient',
           'CORConfigurationClient', 'PipelineSynchronizationServer',
           'PipelineSynchronizationClient', 'PipelineEventServer', 'PipelineEventClient']


import sys
import logging
import functools
import traceback
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO


from .NdpCommon import DATE_FORMAT, FS


def logException(func):
    """
    Decorator for wrapping a function call and catching any exception thrown so
    that it can go into the current logging instance.
    """
    
    logger = logging.getLogger('__main__')
    
    @functools.wraps(func)
    def tryExceptWrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            try:
                logger.error("%s in %s failed with %s at line %i", func, func.func_code.co_filename, str(e), func.func_code.co_firstlineno + 1)
            except AttributeError:
                logger.error("%s in %s failed with %s at line %i", func, func.__code__.co_filename, str(e), func.__code__.co_firstlineno + 1)
            
            # Grab the full traceback and save it to a string via StringIO
            fileObject = StringIO()
            traceback.print_tb(exc_traceback, file=fileObject)
            tbString = fileObject.getvalue()
            fileObject.close()
            
            # Print the traceback to the logger as a series of DEBUG messages
            for line in tbString.split('\n'):
                logger.debug("%s", line)
                
    return tryExceptWrapper


class PipelineMessageServer(object):
    """
    Class for broadcasting configuration information to the different pipelines 
    running on various servers.  This is implemented using 0MQ in a PUB/SUB
    scheme.
    """
    
    def __init__(self, addr=('ndp', 5832), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind('tcp://*:%i' % addr[1])
        
    def packetStartTime(self, utcStartTime):
        """
        Send the UTC start time to TBN and DRX clients.
        """
        
        try:
            utcStartTime = utcStartTime.strftime(DATE_FORMAT)
        except AttributeError:
            pass
        self.socket.send_string('UTC %s' % utcStartTime)
        
    def drxConfig(self, beam, tuning, frequency, filter, gain, subslot):
        """
        Send DRX configuration information out to the clients.  This 
        includes:
          * the beam number this update applied to
          * the tuning number this update applied to
          * frequency in Hz
          * filter code
          * gain setting
          * execution subslot
        """
        
        self.socket.send_string('DRX %i %i %.6f %i %i %i' % (beam, tuning, frequency, filter, gain, subslot))
        
    def bamConfig(self, beam, delays, gains, subslot):
        """
        Send BAM configuration information out to the clients.  This includes:
          * the beam number this update applies to
          * the delays as a 1-D numpy array
          * the gains as a 3-D numpy array
          * the subslot in which the configuration is implemented
        """
        
        bDelays = binascii.hexlify( delays.tostring() ).decode()
        bGains = binascii.hexlify( gains.tostring() ).decode()
        self.socket.send_string('BAM %i %s %s %i' % (beam, bDelays, bGains, subslot))
        
    def corConfig(self, navg, gain, subslot):
        """
        Send COR configuration information out the clients.  This includes:
          * the integration time in units of subslots
          * the gain
          * the subslot in which the configuration is implemented
        """
        
        self.socket.send_string('COR %i %i %i' % (navg, gain, subslot))
        
    def trigger(self, trigger, samples, mask, local=False):
        """
        Send a trigger to start dumping TBF data.  This includes:
          * the trigger time
          * the number of samples to dump
          * the DRX tuning mask to use
          * whether or not to dump to disk
        """
        
        self.socket.send_string('TRIGGER %i %i %i %i' % (trigger, samples, mask, local))
        
    def close(self):
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineMessageClient(object):
    """
    Client side version of PipelineMessageServer that is used to collect the 
    configuration updates as they come in.
    """
    
    def __init__(self, group, addr=('ndp', 5832), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.SUB)
        try:
            self.socket.setsockopt(zmq.SUBSCRIBE, group)
        except TypeError:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, group)
        self.socket.connect('tcp://%s:%i' % addr)
        
    def __call__(self, block=False):
        """
        Pull in information if it is available.  If a message from the server
        is available it is returned.  Otherwise the behavior is determined by
        the 'block' keyword.  If 'block' is True, the function blocks until a 
        message is received.  If 'block' is False, False is returned 
        immediately.
        """
        
        try:
            msg = self.socket.recv_string(flags=(0 if block else zmq.NOBLOCK))
            return msg
        except zmq.error.ZMQError:
            return False
            
    def close(self):
        """
        Close out the client.
        """
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class StartTimeClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for receiving 
    start time information."""
    
    def __init__(self, addr=('ndp', 5832), context=None):
        super(StartTimeClient, self).__init__('UTC', addr=addr, context=context)
        
    def __call__(self):
        msg = super(StartTimeClient, self).__call__(block=True)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields = msg.split(None, 1)
            start = datetime.datetime.strptime(fields[1], DATE_FORMAT)
            return start


class TriggerClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for trigger 
    information.
    """
    
    def __init__(self, addr=('ndp', 5832), context=None):
        super(TriggerClient, self).__init__('TRIGGER', addr=addr, context=context)
        
    def __call__(self, block=False):
        msg = super(TriggerClient, self).__call__(block=block)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields  = msg.split(None, 4)
            trigger = int(fields[1], 10)
            samples = int(fields[2], 10)
            mask    = int(fields[3], 10)
            local   = bool(int(fields[4], 10))
            return trigger, samples, mask, local


class DRXConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for DRX 
    configuration information.
    """
    
    def __init__(self, addr=('ndp', 5832), context=None):
        super(DRXConfigurationClient, self).__init__('DRX', addr=addr, context=context)
        
    def __call__(self):
        msg = super(DRXConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields    = msg.split(None, 6)
            beam      = int(fields[1], 10)
            tuning    = int(fields[2], 10)
            frequency = float(fields[3])
            filter    = int(fields[4], 10)
            gain      = int(fields[5], 10)
            subslot   = int(fields[6], 10)
            return beam, tuning, frequency, filter, gain, subslot


class BAMConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for BAM 
    configuration information.
    """
    
    def __init__(self, addr=('ndp', 5832), context=None):
        super(BAMConfigurationClient, self).__init__('BAM', addr=addr, context=context)
        
    def __call__(self):
        msg = super(BAMConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields = msg.split(None, 5)
            beam = int(fields[1], 10)
            delays = numpy.fromstring( binascii.unhexlify(fields[2]), dtype='>H' )
            delays.shape = (512,)
            gains = numpy.fromstring( binascii.unhexlify(fields[3]), dtype='>H' )
            gains.shape = (256,2,2)
            subslot = int(fields[4], 10)
            return beam, delays, gains, subslot


class CORConfigurationClient(PipelineMessageClient):
    """
    Sub-class of PipelineMessageClient that is specifically for COR 
    configuration information.
    """
    
    def __init__(self, addr=('ndp', 5832), context=None):
        super(CORConfigurationClient, self).__init__('COR', addr=addr, context=context)
        
    def __call__(self):
        msg = super(CORConfigurationClient, self).__call__(block=False)
        if not msg:
            # Nothing to report
            return False
        else:
            # Unpack
            fields  = msg.split(None, 3)
            navg    = int(fields[0], 10)
            gain    = int(fields[1], 10)
            subslot = int(fields[2], 10)
            return navg, gain, subslot


class PipelineSynchronizationServer(object):
    """
    Class to provide packet-level synchronization across the pipelines.  
    This uses 0MQ in a ROUTER/DEALER setup to make sure clients reach the
    same point at the same time.
    """
    
    def __init__(self, nClients=6, addr=('ndp', 5833), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind('tcp://*:%i' % addr[1])
        
        # Setup the client count
        self.nClients = nClients
        
        # Setup the threading
        self.thread = None
        self.alive = threading.Event()
        
    def start(self):
        """
        Start the synchronization pool.
        """
        
        if self.thread is not None:
            self.stop()
            
        self.thread = threading.Thread(target=self._sync, name='synchronizer')
        self.thread.setDaemon(1)
        self.alive.set()
        self.thread.start()
        
    def stop(self):
        """
        Stop the synchronization pool.
        """
        
        if self.thread is not None:
            self.alive.clear()
            self.thread.join()
            
            self.thread = None
            
    def _sync(self):
        clients = []
        nAct = 0
        nRecv = 0
        
        while self.alive.isSet():
            client, msg = self.socket.recv_multipart()
            try:
                msg = msg.decode()
            except AttributeError:
                # Python2 catch
                pass
                
            if msg == 'JOIN':
                if client not in clients:
                    clients.append( client )
                    nAct += 1
                    print("FOUND '%s'" % client)
                    
            elif msg == 'LEAVE':
                try:
                    del clients[clients.index(client)]
                    nAct -= 1
                    print("LOST '%s'" % client)
                except ValueError:
                    pass
                    
            elif msg[:3] == 'TAG':
                nRecv += 1
                
                if nRecv == nAct:
                    for client in clients:
                        self.socket.send_multipart([client, msg])
                        
    def close(self):
        """
        Stop the synchronization pool and close out the server.
        """
        
        self.stop()
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineSynchronizationClient(object):
    """
    Client class for PipelineSynchronizationClient.
    """
    
    def __init__(self, id=None, addr=('ndp', 5833), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.DEALER)
        if id is not None:
            try:
                self.socket.setsockopt(zmq.IDENTITY, str(id))
            except TypeError:
                self.socket.setsockopt_string(zmq.IDENTITY, str(id))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect('tcp://%s:%i' % addr)
        
        # Connect to the server
        self.socket.send_string('JOIN')
        
    def __call__(self, tag=None):
        self.socket.send_string('TAG:%s' % tag)
        return self.socket.recv_string()
        
    def close(self):
        """
        Leave the synchronization pool and close out the client.
        """
        
        self.socket.send_string('LEAVE')
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineEventServer(object):
    """
    Class to provide a distributed event across the pipelines.  This uses 
    0MQ in a REQUEST/REPLY setup to make sure clients can lock/unlock each
    other to control data flow.
    """
    
    def __init__(self, addr=('ndp', 5834), context=None, timeout=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind('tcp://*:%i' % addr[1])
        
        # Setup the poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        # Setup the event
        self.timeout = timeout
        self._state = {}
        
        # Setup the threading
        self.thread = None
        self.alive = threading.Event()
        
    def _set(self, id):
        self._state[id] = time.time()
        return True
        
    def _is_set(self, id):
        if len(self._state):
            if self.timeout is None:
                return True
                
            else:
                for id in sorted(self._state):
                    age = time.time() - self._state[id]
                    if age <= self.timeout:
                        return True
                    else:
                        self._clear(id)
                        
        return False
        
    def _clear(self, id):
        try:
            del self._state[id]
            return True
            
        except KeyError:
            return False
            
    def start(self):
        """
        Start the event pool.
        """
        
        if self.thread is not None:
            self.stop()
            
        self._state = {}
        
        self.thread = threading.Thread(target=self._event, name='event')
        self.thread.setDaemon(1)
        self.alive.set()
        self.thread.start()
        
    def stop(self):
        """
        Stop the event pool.
        """
        
        if self.thread is not None:
            self.alive.clear()
            self.thread.join()
            
            self.thread = None
            
    def _event(self):
        while self.alive.isSet():
            msg = dict(self.poller.poll(1000))
            if msg:
                if msg.get(self.socket) == zmq.POLLIN:
                    msg = self.socket.recv_string(zmq.NOBLOCK)
                    id, msg = msg.split(None, 1)
                    
                    if msg == 'SET':
                        status = self._set(id)
                    elif msg == 'CLEAR':
                        status = self._clear(id)
                    elif msg == 'IS_SET':
                        status = self._is_set(id)
                    elif msg == 'LEAVE':
                        status = self._clear(id)
                    else:
                        status = False
                    self.socket.send_string(str(status))
                    
    def close(self):
        """
        Stop the locking pool and close out the server.
        """
        
        self.stop()
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class PipelineEventClient(object):
    """
    Client class for PipelineEventClient.
    """
    
    def __init__(self, id=None, addr=('ndp', 5834), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.REQ)
        if id is None:
            id = uuid4()
        try:
            self.socket.setsockopt(zmq.IDENTITY, str(id))
        except TypeError:
            self.socket.setsockopt_string(zmq.IDENTITY, str(id))
        self.socket.setsockopt(zmq.LINGER, 100)
        self.socket.connect('tcp://%s:%i' % addr)
        
        # Save the ID
        self.id = self.socket.getsockopt(zmq.IDENTITY)
        try:
            self.id = self.id.decode()
        except AttributeError:
            # Python2 catch
            pass
            
    def is_set(self):
        self.socket.send_string('%s %s' % (self.id, 'IS_SET'))
        return True if self.socket.recv_string() == 'True' else False
        
    def isSet(self):
        return self.is_set()
        
    def set(self):
        self.socket.send_string('%s %s' % (self.id, 'SET'))
        return True if self.socket.recv_string() == 'True' else False
        
    def clear(self):
        self.socket.send_string('%s %s' % (self.id, 'CLEAR'))
        return True if self.socket.recv_string() == 'True' else False
        
    def wait(self, timeout=None):
        t0 = time.time()
        while not self.is_set():
            time.sleep(0.01)
            t1 = time.time()
            if timeout is not None:
                if t1-t0 > timeout:
                    return False
        return True
        
    def close(self):
        """
        Leave the synchronization pool and close out the client.
        """
        
        self.socket.send_string('%s %s' % (self.id, 'LEAVE'))
        status = True if self.socket.recv_string() == 'True' else False
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()
            
        return status


class InternalTrigger(object):
    """
    Class for signaling that a potentially interesting event has occurred.  
    This is sent to the InternalTrigger server for validation.  This is 
    implemented using 0MQ in a PUSH/PULL scheme.
    """
    
    def __init__(self, id=None, addr=('ndp', 5835), context=None):
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.PUSH)
        if id is None:
            id = uuid4()
        try:
            self.socket.setsockopt(zmq.IDENTITY, str(id))
        except TypeError:
            self.socket.setsockopt_string(zmq.IDENTITY, str(id))
        self.socket.setsockopt(zmq.LINGER, 10)
        self.socket.connect('tcp://%s:%i' % addr)
        
        # Save the ID
        self.id = self.socket.getsockopt(zmq.IDENTITY)
        self.id = self.id.decode()
        
    def __call__(self, timestamp):
        """
        Send the event's timestamp as a DP/ADP/NDP timestamp value, i.e., 
        int(UNIX time * 196e6).
        """
        
        self.socket.send_string('%s %s' % (self.id, str(timestamp)))
        
    def close(self):
        """
        Leave the triggering pool and close out the client.
        """
        
        self.socket.close()
        if self.newContext:
            self.context.destroy()


class InternalTriggerProcessor(object):
    """
    Class to gather triggers from various InternalTrigger clients, validate 
    them, and actually act on the trigger.
    """
    
    def __init__(self, port=5835, coincidence_window=5e-4, min_coincident=4, deadtime=60.0, callback=None, context=None):
        # Set the port to use
        self.port = port
        
        # Set the coincidence window time limit (window size used to determine 
        # if the triggers occurred at the same time)
        self.coincidence_window = int(float(coincidence_window)*FS)
        
        # Set the minimum number of coincident events within the time window to 
        # accept as real events
        self.min_coincident = int(min_coincident)
        
        # Set the deadtime (required downtime between valid triggers) and 
        # callback (function to call on a valid trigger)
        self.deadtime = int(float(deadtime)*FS)
        self.callback = callback
        
        # Create the context
        if context is not None:
            self.context = context
            self.newContext = False
        else:
            self.context = zmq.Context()
            self.newContext = True
            
        # Create the event list
        self.events = {}
        
        # Setup the thread control
        self.shutdown_event = threading.Event()
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def run(self):
        tLast = 0
        
        # Create the socket and configure it
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 10)
        self.socket.bind('tcp://*:%i' % self.port)
        
        # Setup the poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        while not self.shutdown_event.is_set():
            # Get an event and parse it out
            msg = dict(self.poller.poll(5000))
            if msg:
                if msg.get(self.socket) == zmq.POLLIN:
                    msg = self.socket.recv_string(zmq.NOBLOCK)
                    try:
                        id, timestamp = msg.split(None, 1)
                        timestamp = int(timestamp, 10)
                    except ValueError:
                        continue
                        
                    # Ignore events that occurring during the mandatory deadtime
                    if timestamp - tLast < self.deadtime:
                        continue
                        
                    # Store the event
                    self.events[id] = timestamp
                    
                    # Validate the event(s)
                    count = len(self.events)
                    newest = max(self.events.values())
                    oldest = min(self.events.values())
                    diff = newest - oldest
                    if count >= self.min_coincident and diff <= self.coincidence_window:
                        ## Looks like we have an event, update the state and send the 
                        ## trigger out
                        tLast = newest
                        self.events.clear()
                        if self.callback is not None:
                            self.callback(oldest)
                            
        # Close out the socket
        self.socket.close()
        if self.newContext:
            self.context.destroy()
