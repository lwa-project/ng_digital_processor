#!/usr/bin/env python3

from ndp import MCS2 as MCS
from ndp import Ndp
from ndp.NdpCommon import *
from ndp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPVerbsCapture as UDPCapture
from bifrost.packet_writer import HeaderInfo, UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.fft import Fft
from bifrost.fir import Fir
from bifrost.quantize import quantize as Quantize
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

import numpy as np
import signal
import logging
import time
import os
import argparse
import ctypes
import threading
import json
import socket
import struct
#import time
import datetime
import pickle
from collections import deque

#from numpy.fft import ifft
#from scipy import ifft
from scipy.fftpack import ifft
from scipy.signal import get_window as scipy_window, firwin as scipy_firwin

ACTIVE_DRX_CONFIG = threading.Event()

INT_CHAN_BW = 50e3

FILTER2BW = {1:   250000,
             2:   500000,
             3:  1000000,
             4:  2000000,
             5:  4900000,
             6:  9800000,
             7: 19600000}
FILTER2CHAN = {1:   250000//int(INT_CHAN_BW),
               2:   500000//int(INT_CHAN_BW),
               3:  1000000//int(INT_CHAN_BW),
               4:  2000000//int(INT_CHAN_BW),
               5:  4900000//int(INT_CHAN_BW),
               6:  9800000//int(INT_CHAN_BW),
               7: 19600000//int(INT_CHAN_BW)}

__version__    = "0.1"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2023, The Long Wavelenght Array Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

def pfb_window(P):
    win_coeffs = scipy_window("hamming", 4*P)
    sinc       = scipy_firwin(4*P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    win_coeffs /= win_coeffs.max()
    return win_coeffs

#{"nbit": 4, "nchan": 136, "nsrc": 16, "chan0": 1456, "time_tag": 288274740432000000}
class CaptureOp(object):
    def __init__(self, log, *args, **kwargs):
        self.log    = log
        self.args   = args
        self.kwargs = kwargs
        self.nbeam_max = self.kwargs['nbeam_max']
        del self.kwargs['nbeam_max']
        self.shutdown_event = threading.Event()
        ## HACK TESTING
        #self.seq_callback = None
    def shutdown(self):
        self.shutdown_event.set()
    def seq_callback(self, seq0, chan0, nchan, nsrc,
                     time_tag_ptr, hdr_ptr, hdr_size_ptr):
        time_tag = seq0 * 2*NCHAN # spectrum number -> samples
        time_tag_ptr[0] = time_tag
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {'time_tag': time_tag,
               'chan0':    chan0,
               'nsrc':     nsrc,
               'nchan':    nchan,
               'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
               'bw':       nchan*CHAN_BW,
               'nstand':   1,
               'nbeam':    1,
               'npol':     2,
               'complex':  True,
               'nbit':     32}
        print("******** CFREQ:", hdr['cfreq'])
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_ibeam(self.seq_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
        del capture

class ReChannelizerOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=250, nbeam_max=1, pfb_inverter=True, guarantee=True, core=None, gpu=None):
        self.log          = log
        self.iring        = iring
        self.oring        = oring
        self.ntime_gulp   = ntime_gulp
        self.nbeam_max    = nbeam_max
        self.pfb_inverter = pfb_inverter
        self.guarantee    = guarantee
        self.core         = core
        self.gpu          = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        # Setup the PFB inverter
        ## Metadata
        nbeam, npol = self.nbeam_max, 2
        ## PFB data arrays
        self.fdata = BFArray(shape=(self.ntime_gulp,NCHAN,nbeam*npol), dtype=np.complex64, space='cuda')
        self.gdata = BFArray(shape=(self.ntime_gulp,NCHAN,nbeam*npol), dtype=np.complex64, space='cuda')
        self.gdata2 = BFArray(shape=(self.ntime_gulp,NCHAN,nbeam*npol), dtype=np.complex64, space='cuda')
        ## PFB inversion matrix
        matrix = BFArray(shape=(self.ntime_gulp//4,4,NCHAN,nbeam*npol), dtype=np.complex64)
        self.imatrix = BFArray(shape=(self.ntime_gulp//4,4,NCHAN,nbeam*npol), dtype=np.complex64, space='cuda')
        
        pfb = pfb_window(NCHAN)
        pfb = pfb.reshape(4, -1)
        pfb.shape += (1,)
        pfb.shape = (1,)+pfb.shape
        matrix[:,:4,:,:] = pfb
        matrix = matrix.copy(space='cuda')
        
        pfft = Fft()
        pfft.init(matrix, self.imatrix, axes=1)
        pfft.execute(matrix, self.imatrix, inverse=False)
        
        wft = 0.3
        BFMap(f"""
              a = (a.mag2() / (a.mag2() + {wft}*{wft})) * (1+{wft}*{wft}) / a.conj();
              """,
              {'a':self.imatrix})
        
        self.imatrix = self.imatrix.reshape(-1, 4, NCHAN*nbeam*npol)
        del matrix
        del pfft
        
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        if self.gpu is not None:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("ReChannelizer: Start of new sequence: %s", str(ihdr))
                
                time_tag = ihdr['time_tag']
                nbeam    = ihdr['nbeam']
                chan0    = ihdr['chan0']
                nchan    = ihdr['nchan']
                chan_bw  = ihdr['bw'] / nchan
                npol     = ihdr['npol']
                
                igulp_size = self.ntime_gulp*nchan*nbeam*npol*8        # complex64
                ishape = (self.ntime_gulp,nchan,nbeam*npol)
                self.iring.resize(igulp_size, 5*igulp_size)
                
                ochan = int(round(CLOCK / 2 / INT_CHAN_BW))
                otime_gulp = self.ntime_gulp*NCHAN // ochan
                ogulp_size = otime_gulp*ochan*nbeam*npol*8 # complex64
                oshape = (otime_gulp,ochan,nbeam*npol)
                self.oring.resize(ogulp_size)
                
                ohdr = ihdr.copy()
                ohdr['chan0'] = 0
                ohdr['nchan'] = ochan
                ohdr['bw']    = CLOCK / 2
                ohdr_str = json.dumps(ohdr)
                
                # Zero out self.fdata in case chan0 has changed
                BFMemSet(self.fdata, 0)
                
                with oring.begin_sequence(time_tag=time_tag, header=ohdr_str) as oseq:
                    prev_time = time.time()
                    iseq_spans = iseq.read(igulp_size)
                    for ispan in iseq_spans:
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            idata = ispan.data_view(np.complex64).reshape(ishape)
                            odata = ospan.data_view(np.complex64).reshape(oshape)
                            
                            ### From here until going to the output ring we are on the GPU
                            t0 = time.time()
                            try:
                                copy_array(bdata, idata)
                            except NameError:
                                bdata = idata.copy(space='cuda')
                                
                            # Pad out to the full 98 MHz bandwidth
                            t1 = time.time()
                            BFMap(f"""
                                  a(i,j+{chan0},k) = b(i,j,k);
                                  a(i,j+{chan0},k) = b(i,j,k);
                                  """,
                                  {'a': self.fdata, 'b': bdata},
                                  axis_names=('i','j','k'),
                                  shape=(self.ntime_gulp,nchan,nbeam*npol))
                            
                            ## PFB inversion
                            ### Initial IFFT
                            t2 = time.time()
                            self.gdata = self.gdata.reshape(self.fdata.shape)
                            try:
                                bfft.execute(self.fdata, self.gdata, inverse=True)
                            except NameError:
                                bfft = Fft()
                                bfft.init(self.fdata, self.gdata, axes=1, apply_fftshift=True)
                                bfft.execute(self.fdata, self.gdata, inverse=True)
                                
                            if self.pfb_inverter:
                                ### The actual inversion
                                t4 = time.time()
                                self.gdata = self.gdata.reshape(self.imatrix.shape)
                                try:
                                    pfft.execute(self.gdata, self.gdata2, inverse=False)
                                except NameError:
                                    pfft = Fft()
                                    pfft.init(self.gdata, self.gdata2, axes=1)
                                    pfft.execute(self.gdata, self.gdata2, inverse=False)
                                    
                                BFMap("a *= b / (%i*2)" % NCHAN,
                                      {'a':self.gdata2, 'b':self.imatrix})
                                     
                                pfft.execute(self.gdata2, self.gdata, inverse=True)
                                
                            ## FFT to re-channelize
                            t5 = time.time()
                            self.gdata = self.gdata.reshape(-1, ochan, nbeam*npol)
                            try:
                                ffft.execute(self.gdata, rdata, inverse=False)
                            except NameError:
                                rdata = BFArray(shape=(otime_gulp,ochan,nbeam*npol), dtype=np.complex64, space='cuda')
                                
                                ffft = Fft()
                                ffft.init(self.gdata, rdata, axes=1, apply_fftshift=True)
                                ffft.execute(self.gdata, rdata, inverse=False)
                                
                            ## Save
                            t6 = time.time()
                            copy_array(odata, rdata)
                            
                            t7 = time.time()
                            # print(t7-t0, '->', t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6)
                            
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})
                        
            try:
                del bdata
                del bfft
                del pfft
                del rdata
                del ffft
            except NameError:
                pass

class TEngineOp(object):
    def __init__(self, log, iring, oring, beam=1, ntime_gulp=2500, guarantee=True, core=None, gpu=None):
        self.log        = log
        self.iring      = iring
        self.oring      = oring
        self.beam       = beam
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        self.gpu        = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog   = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self._pending = deque()
        self.gain = [6, 6]
        self.rFreq = [30e6, 60e6]
        self.filt = 7
        self.nchan_out = FILTER2CHAN[self.filt]
        
        coeffs = np.array([ 0.0111580, -0.0074330,  0.0085684, -0.0085984,  0.0070656, -0.0035905, 
                           -0.0020837,  0.0099858, -0.0199800,  0.0316360, -0.0443470,  0.0573270, 
                           -0.0696630,  0.0804420, -0.0888320,  0.0941650,  0.9040000,  0.0941650, 
                           -0.0888320,  0.0804420, -0.0696630,  0.0573270, -0.0443470,  0.0316360, 
                           -0.0199800,  0.0099858, -0.0020837, -0.0035905,  0.0070656, -0.0085984,  
                            0.0085684, -0.0074330,  0.0111580], dtype=np.float64)
        
        # Setup the T-engine
        if self.gpu is not None:
            BFSetGPU(self.gpu)
        ## Metadata
        nbeam, ntune, npol = 1, 2, 2
        ## Coefficients
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, nbeam*ntune*npol, axis=1)
        coeffs.shape = (coeffs.shape[0],nbeam,ntune,npol)
        self.coeffs = BFArray(coeffs, space='cuda')
        ## Phase rotator state
        phaseState = np.array([0,]*ntune, dtype=np.float64)
        self.phaseState = BFArray(phaseState, space='cuda')
        sampleCount = np.array([0,]*ntune, dtype=np.int64)
        self.sampleCount = BFArray(sampleCount, space='cuda')
        
    def updateConfig(self, config, hdr, time_tag, forceUpdate=False):
        global ACTIVE_DRX_CONFIG
        
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Can we act on this configuration change now?
        if config:
            ## Pull out the beam (something unique to DRX/BAM/COR)
            beam = config[0]
            if beam != self.beam:
                return False
                
            ## Set the configuration time - DRX commands are for the specified subslot in the next second
            subslot = config[-1] / 100.0
            config_time = int(time.time()) + 1 + subslot
            
            ## Is this command from the future?
            if pipeline_time < config_time:
                ### Looks like it, save it for later
                self._pending.append( (config_time, config) )
                config = None
                
                ### Is there something pending?
                try:
                    stored_time, stored_config = self._pending[0]
                    if pipeline_time >= stored_time:
                        config_time, config = self._pending.popleft()
                except IndexError:
                    pass
            else:
                ### Nope, this is something we can use now
                pass
                
        else:
            ## Is there something pending?
            try:
                stored_time, stored_config = self._pending[0]
                if pipeline_time >= stored_time:
                    config_time, config = self._pending.popleft()
            except IndexError:
                #print("No pending configuration at %.1f" % pipeline_time)
                pass
                
        if config:
            self.log.info("TEngine: New configuration received for beam %i, tuning %i (delta = %.1f subslots)", config[0], config[1], (pipeline_time-config_time)*100.0)
            beam, tuning, freq, filt, gain, subslot = config
            if beam != self.beam:
                self.log.info("TEngine: Not for this beam, skipping")
                return False
                
            self.rFreq[tuning] = freq
            self.filt = filt
            self.nchan_out = FILTER2CHAN[filt]
            self.gain[tuning] = gain
            
            fDiff = freq - (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW - CHAN_BW / 2
            self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
            
            if self.gpu != -1:
                BFSetGPU(self.gpu)
                
            phaseState = fDiff/(self.nchan_out*CHAN_BW)
            phaseRot = np.exp(-2j*np.pi*phaseState*np.arange(self.ntime_gulp*self.nchan_out, dtype=np.float64))
            phaseRot = phaseRot.astype(np.complex64)
            copy_array(self.phaseState, np.array([phaseState,], dtype=np.float64))
            self.phaseRot = BFAsArray(phaseRot, space='cuda')
            
            ACTIVE_DRX_CONFIG.set()
            
            return True
            
        elif forceUpdate:
            self.log.info("TEngine: New sequence configuration received")
            
            try:
                fDiff = self.rFreq - (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW - CHAN_BW / 2
            except AttributeError:
                self.rFreq = (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW + CHAN_BW / 2
                fDiff = 0.0
            self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
            
            if self.gpu != -1:
                BFSetGPU(self.gpu)
                
            phaseState = fDiff/(self.nchan_out*CHAN_BW)
            phaseRot = np.exp(-2j*np.pi*phaseState*np.arange(self.ntime_gulp*self.nchan_out, dtype=np.float64))
            phaseRot = phaseRot.astype(np.complex64)
            copy_array(self.phaseState, np.array([phaseState,], dtype=np.float64))
            self.phaseRot = BFAsArray(phaseRot, space='cuda')
            
            return False
            
        else:
            return False
            
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        if self.gpu is not None:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
                             
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("TEngine: Start of new sequence: %s", str(ihdr))
                
                self.rFreq[0] = 30e6
                self.rFreq[1] = 60e6
                self.updateConfig( ihdr, iseq.time_tag, forceUpdate=True )
                
                nbeam    = ihdr['nbeam']
                chan0    = ihdr['chan0']
                nchan    = ihdr['nchan']
                chan_bw  = ihdr['bw'] / nchan
                npol     = ihdr['npol']
                ntune    = 2
                
                igulp_size = self.ntime_gulp*nchan*nbeam*npol*8                # complex64
                ishape = (self.ntime_gulp,nchan,nbeam,npol)
                self.iring.resize(igulp_size, 10*igulp_size)
                
                ogulp_size = self.ntime_gulp*self.nchan_out*nbeam*ntune*npol*2       # 8+8 complex
                oshape = (self.ntime_gulp*self.nchan_out,nbeam,ntune,npol)
                self.oring.resize(ogulp_size, 10*ogulp_size)
                
                ticksPerTime = int(FS) // int(INT_CHAN_BW)
                base_time_tag = iseq.time_tag
                sample_count = np.array([0,]*ntune, dtype=np.int64)
                copy_array(self.sampleCount, sample_count)
                
                ohdr = {}
                ohdr['nbeam']   = nbeam
                ohdr['ntune']   = ntune
                ohdr['npol']    = npol
                ohdr['complex'] = True
                ohdr['nbit']    = 4
                ohdr['fir_size'] = self.coeffs.shape[0]
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    ohdr['time_tag'] = base_time_tag
                    ohdr['cfreq0']   = self.rFreq[0]
                    ohdr['cfreq1']   = self.rFreq[1]
                    ohdr['bw']       = self.nchan_out*INT_CHAN_BW
                    ohdr['gain0']    = self.gain[0]
                    ohdr['gain1']    = self.gain[1]
                    ohdr['filter']   = self.filt
                    ohdr_str = json.dumps(ohdr)
                    
                    # Update the channels to pull in
                    tchan0 = int(self.rFreq[0] / INT_CHAN_BW + 0.5) - self.nchan_out//2
                    tchan1 = int(self.rFreq[1] / INT_CHAN_BW + 0.5) - self.nchan_out//2
                    
                    # Adjust the gain to make this ~compatible with LWA1
                    act_gain0 = self.gain[0] + 12
                    act_gain1 = self.gain[1] + 12
                    rel_gain = np.array([1.0, (2**act_gain0)/(2**act_gain1)], dtype=np.float32)
                    rel_gain = BFArray(rel_gain, space='cuda')
                    
                    with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            with oseq.reserve(ogulp_size) as ospan:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                ## Setup and load
                                idata = ispan.data_view(np.complex64).reshape(ishape)
                                odata = ospan.data_view(np.int8).reshape(oshape)
                                
                                ## Prune the data ahead of the IFFT
                                try:
                                    pdata[:,:,:,0,:] = idata[:,tchan0:tchan0+self.nchan_out,:,:]
                                    pdata[:,:,:,1,:] = idata[:,tchan1:tchan1+self.nchan_out,:,:]
                                except NameError:
                                    pshape = (self.ntime_gulp,self.nchan_out,nbeam,ntune,npol)
                                    pdata = BFArray(shape=pshape, dtype=np.complex64, space='cuda_host')
                                    
                                    pdata[:,:,:,0,:] = idata[:,tchan0:tchan0+self.nchan_out,:,:]
                                    pdata[:,:,:,1,:] = idata[:,tchan1:tchan1+self.nchan_out,:,:]
                                    
                                ### From here until going to the output ring we are on the GPU
                                try:
                                    copy_array(bdata, pdata)
                                except NameError:
                                    bdata = pdata.copy(space='cuda')
                                    
                                ## IFFT
                                try:
                                    gdata = gdata.reshape(*bdata.shape)
                                    bfft.execute(bdata, gdata, inverse=True)
                                except NameError:
                                    gdata = BFArray(shape=bdata.shape, dtype=np.complex64, space='cuda')
                                    
                                    bfft = Fft()
                                    bfft.init(bdata, gdata, axes=1, apply_fftshift=True)
                                    bfft.execute(bdata, gdata, inverse=True)
                                    
                                ## Phase rotation and output "desired gain imbalance" correction
                                gdata = gdata.reshape((-1,nbeam*ntune*npol))
                                BFMap("""
                                      auto k = (j / 2);// % 2;
                                      a(i,j) *= exp(Complex<float>(r(k), -2*BF_PI_F*r(k)*fmod(g(k)*s(k), 1.0)))*b(i,k);
                                      """, 
                                      {'a':gdata, 'b':self.phaseRot, 'g':self.phaseState, 's':self.sampleCount, 'r':rel_gain},
                                      axis_names=('i','j'),
                                      shape=gdata.shape, 
                                      extra_code="#define BF_PI_F 3.141592654f")
                                gdata = gdata.reshape((-1,nbeam,ntune,npol))
                                
                                ## FIR filter
                                try:
                                    bfir.execute(gdata, fdata)
                                except NameError:
                                    fdata = BFArray(shape=gdata.shape, dtype=gdata.dtype, space='cuda')
                                    
                                    bfir = Fir()
                                    bfir.init(self.coeffs, 1)
                                    bfir.execute(gdata, fdata)
                                    
                                ## Quantization
                                try:
                                    Quantize(fdata, qdata, scale=8./(2**act_gain0 * np.sqrt(self.nchan_out)))
                                except NameError:
                                    qdata = BFArray(shape=fdata.shape, native=False, dtype='ci8', space='cuda')
                                    Quantize(fdata, qdata, scale=8./(2**act_gain0 * np.sqrt(self.nchan_out)))
                                    
                                ## Save
                                try:
                                    copy_array(tdata, qdata)
                                except NameError:
                                    tdata = qdata.copy('system')
                                odata[...] = tdata.view(np.int8).reshape(self.ntime_gulp*self.nchan_out,nbeam,ntune,npol,2)
                                
                            ## Update the base time tag
                            base_time_tag += self.ntime_gulp*ticksPerTime
                            
                            ## Update the sample counter
                            sample_count += oshape[0]
                            copy_array(self.sampleCount, sample_count)
                            
                            ## Check for an update to the configuration
                            if self.updateConfig( ihdr, base_time_tag, forceUpdate=False ):
                                reset_sequence = True
                                sample_count *= 0
                                copy_array(self.sampleCount, sample_count)
                                
                                ### New output size/shape
                                ngulp_size = self.ntime_gulp*self.nchan_out*nbeam*ntune*npol*2               # 8+8 complex
                                nshape = (self.ntime_gulp*self.nchan_out,nbeam,ntune,npol)
                                if ngulp_size != ogulp_size:
                                    ogulp_size = ngulp_size
                                    oshape = nshape
                                    
                                    self.oring.resize(ogulp_size)
                                    
                                ### Clean-up
                                try:
                                    del pdata
                                    del bdata
                                    del gdata
                                    del bfft
                                    del fdata
                                    del bfir
                                    del qdata
                                    del tdata
                                except NameError:
                                    pass
                                    
                                break
                                
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                                                 
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        ## Clean-up
                        try:
                            del pdata
                            del bdata
                            del gdata
                            del fdata
                            del qdata
                            del tdata
                        except NameError:
                            pass
                            
                        break

class PacketizeOp(object):
    def __init__(self, log, iring, osock, beam0=1, npkt_gulp=128, nbeam_max=1, ntune_max=2, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.osock      = osock
        self.beam0      = beam0
        self.npkt_gulp  = npkt_gulp
        self.nbeam_max  = nbeam_max
        self.ntune_max  = ntune_max
        self.guarantee  = guarantee
        self.core       = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
    def main(self):
        global FILE_QUEUE
        
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        ntime_pkt     = DRX_NSAMPLE_PER_PKT
        ntime_gulp    = self.npkt_gulp * ntime_pkt
        ninput_max    = self.nbeam_max * self.ntune_max * 2
        igulp_size_max = ntime_gulp * ninput_max * 2 * 2
        
        self.size_proclog.update({'nseq_per_gulp': ntime_gulp})
        
        with UDPTransmit('drx', sock=self.osock, core=self.core) as udt:
            desc0 = HeaderInfo()
            desc1 = HeaderInfo()
            
            was_active = False
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("Writer: Start of new sequence: %s", str(ihdr))
                
                time_tag = ihdr['time_tag']
                cfreq0   = ihdr['cfreq0']
                cfreq1   = ihdr['cfreq1']
                bw       = ihdr['bw']
                gain0    = ihdr['gain0']
                gain1    = ihdr['gain1']
                filt     = ihdr['filter']
                nbeam    = ihdr['nbeam']
                ntune    = ihdr['ntune']
                npol     = ihdr['npol']
                fdly     = (ihdr['fir_size'] - 1) / 2.0
                time_tag0 = iseq.time_tag
                time_tag  = time_tag0
                igulp_size = ntime_gulp*nbeam*ntune*npol
                
                # Figure out where we need to be in the buffer to be at a frame boundary
                NPACKET_SET = 16
                ticksPerSample = int(FS) // int(bw)
                toffset = int(time_tag0) // ticksPerSample
                soffset = toffset % (NPACKET_SET*int(ntime_pkt))
                if soffset != 0:
                    soffset = NPACKET_SET*ntime_pkt - soffset
                boffset = soffset*nbeam*ntune*npol*2
                print('!!', '@', self.beam0, toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset, ' at ', ticksPerSample)
                
                time_tag += soffset*ticksPerSample                  # Correct for offset
                time_tag -= int(round(fdly*ticksPerSample))         # Correct for FIR filter delay
                
                prev_time = time.time()
                desc0.set_decimation(int(FS)//int(bw))
                desc1.set_decimation(int(FS)//int(bw))
                desc0.set_tuning(int(round(cfreq0 / FS * 2**32)))
                desc1.set_tuning(int(round(cfreq1 / FS * 2**32)))
                desc_src = ((1&0x7)<<3)
                
                for ispan in iseq.read(igulp_size, begin=boffset):
                    if ispan.size < igulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    shape = (-1,nbeam,ntune,npol)
                    data = ispan.data_view('ci8').reshape(shape)
                    
                    data0 = data[:,:,0,:].reshape(-1,ntime_pkt,nbeam*npol).transpose(0,2,1).copy()
                    data1 = data[:,:,1,:].reshape(-1,ntime_pkt,nbeam*npol).transpose(0,2,1).copy()
                    
                    for t in range(0, data0[0].shape[0], NPACKET_SET):
                        time_tag_cur = time_tag + t*ticksPerSample*ntime_pkt
                        
                        if ACTIVE_DRX_CONFIG.is_set():
                            if not self.tbfLock.is_set():
                                try:
                                    udt.send(desc0, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+self.beam0, 128, 
                                             data0[t:t+NPACKET_SET,:,:])
                                    udt.send(desc1, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+8+self.beam0, 128, 
                                             data1[t:t+NPACKET_SET,:,:])
                                except Exception as e:
                                    print(type(self).__name__, 'Sending Error', str(e))
                                        
                    time_tag += int(ntime_gulp)*ticksPerSample
                    
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': -1, 
                                              'process_time': process_time,})

def get_utc_start(shutdown_event=None):
    got_utc_start = False
    while not got_utc_start:
        if shutdown_event is not None:
            if shutdown_event.is_set():
                raise RuntimeError("Shutting down without getting the start time")
                
        try:
            with MCS.Communicator() as ndp_control:
                utc_start = ndp_control.report('UTC_START')
                # Check for valid timestamp
                utc_start_tt = int(utc_start, 10)
            got_utc_start = True
        except Exception as ex:
            print(ex)
            time.sleep(1)
    #print("UTC_START:", utc_start)
    #return utc_start
    return utc_start_tt

def get_numeric_suffix(s):
    i = 0
    while True:
        if len(s[i:]) == 0:
            raise ValueError("No numeric suffix in string '%s'" % s)
        try: return int(s[i:])
        except ValueError: i += 1

def main(argv):
    parser = argparse.ArgumentParser(description='LWA-NA NDP T-Engine Service')
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-b', '--beam',       default=1, type=int,       help='DRX beam (1, 2, 3, or 4)')
    parser.add_argument('-p', '--pfb-inverter', action='store_true',     help='Enable the PFB inverter')
    parser.add_argument('-c', '--configfile', default='ndp_config.json', help='Specify config file')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    args = parser.parse_args()
    beam = args.beam
    
    # Fork, if requested
    if args.fork:
        stderr = '/tmp/%s_%i.stderr' % (os.path.splitext(os.path.basename(__file__))[0], tuning)
        daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)
        
    config = Ndp.parse_config_file(args.configfile)
    ntuning = 4
    drxConfigs = config['drx']
    drxConfig = drxConfigs[0]
    
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    if args.logfile is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = Ndp.NdpFileHandler(config, args.logfile)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    verbosity = args.verbose - args.quiet
    if   verbosity >  0: log.setLevel(logging.DEBUG)
    elif verbosity == 0: log.setLevel(logging.INFO)
    elif verbosity <  0: log.setLevel(logging.WARNING)
    
    log.info("Starting %s with PID %i", argv[0], os.getpid())
    log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
    log.info("Version:      %s", __version__)
    log.info("Current MJD:  %f", Ndp.MCS2.slot2mjd())
    log.info("Current MPM:  %i", Ndp.MCS2.slot2mpm())
    log.info("Config file:  %s", args.configfile)
    log.info("Log file:     %s", args.logfile)
    log.info("Dry run:      %r", args.dryrun)
    
    ops = []
    shutdown_event = threading.Event()
    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
        except IndexError:
            pass
        shutdown_event.set()
    for sig in [signal.SIGHUP,
                signal.SIGINT,
                signal.SIGQUIT,
                signal.SIGTERM,
                signal.SIGTSTP]:
        signal.signal(sig, handle_signal_terminate)
    
    log.info('Waiting to get correlator UTC_START timetag')
    utc_start_tt = get_utc_start(shutdown_event)
    utc_start_dt = datetime.datetime.utcfromtimestamp(utc_start_tt/FS)
    log.info("UTC_START: %i = %s", utc_start_tt, utc_start_dt.strftime(DATE_FORMAT))
    
    hostname = socket.gethostname()
    try:
        server_idx = get_numeric_suffix(hostname) - 1
    except ValueError:
        server_idx = 0 # HACK to allow testing on head node "ndp"
    log.info("Hostname:     %s", hostname)
    log.info("Server index: %i", server_idx)
    
    # Network - input
    tengine_idx  = drxConfig['tengine_idx'][beam]
    tngConfig    = config['tengine'][tengine_idx]
    iaddr        = config['host']['tengines'][tengine_idx]
    iport        = config['server']['data_ports' ][tngConfig['pipeline_idx']] + beam + 1
    # Network - output
    recorder_idx = tngConfig['recorder_idx']
    recConfig    = config['recorder'][recorder_idx]
    oaddr        = recConfig['host']
    oport        = recConfig['port']
    obw          = recConfig['max_bytes_per_sec']
        
    nserver = len(config['host']['servers'])
    server0 = 0
    nbeam = drxConfig['beam_count']
    cores = tngConfig['cpus']
    gpus  = tngConfig['gpus']*len(cores)
    pfb_inverter = True
    if 'pfb_inverter' in tngConfig:
        pfb_inverter = tngConfig['pfb_inverter']
        
    log.info("Src address:  %s:%i", iaddr, iport)
    try:
        for b,a,p in zip(range(len(oaddr)), oaddr, oport):
            bstat = ''
            if b >= nbeam:
                bstat = ' (not used)'
            log.info("Dst address:  %i @ %s:%i%s", b, a, p, bstat)
    except TypeError:
        log.info("Dst address:  %s:%i", oaddr, oport)
    log.info("Servers:      %i-%i", server0+1, server0+nserver)
    log.info("Beam:         %i (of %i)", beam+1, nbeam)
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))
    log.info("PFB inverter: %s", str(pfb_inverter))
    
    iaddr = Address(iaddr, iport)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 0.5
    
    capture_ring = Ring(name="capture-%i" % beam, space="cuda_host")
    rechan_ring = Ring(name="rechan-%i" % beam, space="cuda_host")
    tengine_ring = Ring(name="tengine-%i" % beam, space="system")
    
    GSIZE = 2500
    nchan_max = int(round(sum([c['capture_bandwidth'] for c in drxConfigs])/CHAN_BW))    # Subtly different from what is in ndp_drx.py
    
    ops.append(CaptureOp(log, fmt="ibeam1", sock=isock, ring=capture_ring,
                         nsrc=ntuning, src0=server0, max_payload_size=6500,
                         nbeam_max=nbeam, 
                         buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0)))
    ops.append(ReChannelizerOp(log, capture_ring, rechan_ring, ntime_gulp=GSIZE,
                               pfb_inverter=args.pfb_inverter,
                               core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(TEngineOp(log, rechan_ring, tengine_ring,
                         beam=beam, ntime_gulp=GSIZE, 
                         core=cores.pop(0), gpu=gpus.pop(0)))
    raddr = Address(oaddr, oport)
    rsock = UDPSocket()
    rsock.connect(raddr)
    ops.append(PacketizeOp(log, tengine_ring,
                           osock=rsock,
                           nbeam_max=1, beam0=beam+1,
                           npkt_gulp=32, core=cores.pop(0)))
    
    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    for thread in threads:
        #thread.daemon = True
        thread.start()
    while not shutdown_event.is_set():
        signal.pause()
    log.info("Shutdown, waiting for threads to join")
    for thread in threads:
        thread.join()
    log.info("All done")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
