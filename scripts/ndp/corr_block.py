import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.libbifrost import _bf
import bifrost.affinity as cpu_affinity
from bifrost.ring import WriteSpan
from bifrost.device import set_device as BFSetGPU

from btcc import Btcc

import time
import ujson as json
import numpy as np

from lwa352_pipeline.blocks.block_base import Block

class Corr(Block):
    """
    **Functionality**

    This block reads data from a GPU-side bifrost ring buffer and feeds
    it to TCC for correlation, outputing results to another GPU-side buffer.

    **New Sequence Condition**

    This block starts a new sequence each time a new integration
    configuration is loaded or the upstream sequence changes.

    **Input Header Requirements**

    This block requires that the following header fields
    be provided by the upstream data source:

    .. table::
        :widths: 25 10 10 55

        +-----------+--------+-------+------------------------------------------------+
        | Field     | Format | Units | Description                                    |
        +===========+========+=======+================================================+
        | ``seq0``  | int    |       | Spectra number for the first sample in the     |
        |           |        |       | input sequence                                 |
        +-----------+--------+-------+------------------------------------------------+

    **Output Headers**

    This block passes headers from the upstream block with
    the following modifications:

    .. table::
        :widths: 25 10 10 55

        +------------------+----------------+---------+-------------------------------+
        | Field            | Format         | Units   | Description                   |
        +==================+================+=========+===============================+
        | ``seq0``         | int            |         | Spectra number for the        |
        |                  |                |         | *first* sample in the         |
        |                  |                |         | integrated output             |
        +------------------+----------------+---------+-------------------------------+
        | ``acc_len``      | int            |         | Number of spectra integrated  |
        |                  |                |         | into each output sample by    |
        |                  |                |         | this block                    |
        +------------------+----------------+---------+-------------------------------+
        | ``ant_to_input`` | list of ints   |         | This header is removed from   |
        |                  |                |         | the sequence                  |
        +------------------+----------------+---------+-------------------------------+
        | ``input_to_ant`` | list of ints   |         | This header is removed from   |
        |                  |                |         | the sequence                  |
        +------------------+----------------+---------+-------------------------------+

    **Data Buffers**

    *Input Data Buffer*: A GPU-side bifrost ring buffer of 4+4 bit
    complex data in order: ``time x channel x stand x polarization``.

    Each gulp of the input buffer reads ``ntime_gulp`` samples.

    *Output Data Buffer*: A GPU-side bifrost ring buffer of 32+32 bit complex
    integer data. This buffer is in the TCC triangular matrix order:
    ``channel x baseline x complexity``.

    The output buffer is written in single accumulation blocks (an integration of
    ``acc_len`` input time samples).

    **Instantiation**

    :param log: Logging object to which runtime messages should be
        emitted.
    :type log: logging.Logger

    :param iring: bifrost input data ring. This should be on the GPU.
    :type iring: bifrost.ring.Ring

    :param oring: bifrost output data ring. This should be on the GPU.
    :type oring: bifrost.ring.Ring

    :param guarantee: If True, read data from the input ring in blocking "guaranteed"
        mode, applying backpressure if necessary to ensure no data are missed by this block.
    :type guarantee: Bool

    :param core: CPU core to which this block should be bound. A value of -1 indicates no binding.
    :type core: int

    :param gpu: GPU device which this block should target. A value of -1 indicates no binding
    :type gpu: int

    :param ntime_gulp: Number of time samples to copy with each gulp.
    :type ntime_gulp: int

    :param nchan: Number of frequency channels per time sample.
    :type nchan: int

    :param nstand: Number of stands per time sample.
    :type nstand: int

    :param npol: Number of polarizations per stand.
    :type npol: int

    :param acc_len: Accumulation length per output buffer write. This should
        be an integer multiple of the input gulp size ``ntime_gulp``.
        This parameter can be updated at runtime.
    :type acc_len: int

    :parameter etcd_client: Etcd client object used to facilitate control of this block.
        If ``None``, do not use runtime control.
    :type etcd_client: etcd3.client.Etcd3Client

    :parameter autostartat: The start time at which the correlator should
        automatically being correlating without intervention of the runtime control
        system. Use the value ``-1`` to cause integration to being on the next
        gulp.
    :type autostartat: int

    :parameter ant_to_input: an [nstand, npol] list of input IDs used to map
        stand/polarization ``S``, ``P`` to a correlator input. This allows the block
        to pass this information to downstream processors. *This functionality is
        currently unused*
    :type ant_to_input: nstand x npol list of ints

    **Runtime Control and Monitoring**

    This block accepts the following command fields:

    .. table::
        :widths: 25 10 10 55

        +-----------------+--------+---------+------------------------------+
        | Field           | Format | Units   | Description                  |
        +=================+========+=========+==============================+
        | ``acc_len``     | int    | samples | Number of samples to         |
        |                 |        |         | accumulate. This should be a |
        |                 |        |         | multiple of ``ntime_gulp``   |
        +-----------------+--------+---------+------------------------------+
        | ``start_time``  | int    | samples | The desired first time       |
        |                 |        |         | sample in an accumulation.   |
        |                 |        |         | This should be a multiple of |
        |                 |        |         | ``ntime_gulp``, and should   |
        |                 |        |         | be related to GPS time       |
        |                 |        |         | through external knowledge   |
        |                 |        |         | of the spectra count origin  |
        |                 |        |         | (aka SNAP *sync time*). The  |
        |                 |        |         | special value ``-1`` can be  |
        |                 |        |         | used to force an immediate   |
        |                 |        |         | start of the correlator on   |
        |                 |        |         | the next input gulp.         |
        +-----------------+--------+---------+------------------------------+

    """

    def __init__(self, log, iring, oring, ntime_gulp=2500,
                 guarantee=True, core=-1, nchan=192, npol=2, nstand=352, acc_len=2400, gpu=-1, etcd_client=None, autostartat=0, ant_to_input=None):
        assert (acc_len % ntime_gulp == 0), "Acculmulation length must be a multiple of gulp size"
        super(Corr, self).__init__(log, iring, oring, guarantee, core, etcd_client=etcd_client)
        # TODO: Other things we could check:
        # - that nchan/pols/gulp_size matches XGPU compilation
        self.ntime_gulp = ntime_gulp
        self.nchan = nchan
        self.npol = npol
        self.nstand = nstand
        self.matlen = nchan * nstand*(nstand+1)//2 * npol*npol
        self.gpu = gpu

        if self.gpu != -1:
            BFSetGPU(self.gpu)
        
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.igulp_size = self.ntime_gulp*nchan*nstand*npol*1        # complex8
        self.ogulp_size = self.matlen * 8 # complex64

        self.define_command_key('start_time', type=int, initial_val=autostartat,
                                condition=lambda x: (x == -1) or (x % self.ntime_gulp == 0))
        self.define_command_key('acc_len', type=int, initial_val=acc_len,
                                condition=lambda x: x % self.ntime_gulp == 0)
        self.update_stats({'xgpu_acc_len': self.ntime_gulp})

        # initialize TCC. Arrays passed as inputs don't really matter here
        # but we need to pass something
        self.bfcc = Btcc()
        self.bfcc.init(4, int(np.ceil((self.ntime_gulp/16.0))*16), nchan, nstand, npol)
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})

        self.oring.resize(self.ogulp_size)
        time_tag = 1
        self.update_stats({'state': 'starting'})
        with self.oring.begin_writing() as oring:
            prev_time = time.time()
            for iseq in self.iring.read(guarantee=self.guarantee):
                self.log.info('CORR >> new input sequence!')
                process_time = 0
                oseq = None
                ospan = None
                start = False
                # Reload commands on each new sequence
                self.update_pending = True
                self.log.debug("Correlating output")
                ihdr = json.loads(iseq.header.tostring())
                this_gulp_time = ihdr['seq0']
                ohdr = ihdr.copy()
                
                # Remove ant-to-input maps. This block outputs xGPU-formatted data,
                # Which isn't trivially an nstand/npol x nstand/npol array.
                # Assume that the downstream code knows how the baseline list is formatted.
                # It would be nice to put that information in the header, but this seems
                # to cause unexpectedly severe slowdown
                if 'ant_to_input' in ihdr:
                    ohdr.pop('ant_to_input')
                if 'input_to_ant' in ihdr:
                    ohdr.pop('input_to_ant')
                self.sequence_proclog.update(ohdr)
                for ispan in iseq.read(self.igulp_size):
                    if ispan.size < self.igulp_size:
                        self.log.info("CORR >>> Ignoring final gulp (expected %d bytes but got %d)" % (self.igulp_size, ispan.size))
                        continue # ignore final gulp
                    if self.update_pending:
                        self.update_command_vals()
                        # Use start_time = -1 as a special condition to start on the next sample
                        # which is a multiple of the accumulation length
                        acc_len = self.command_vals['acc_len']
                        if self.command_vals['start_time'] == -1:
                            start_time = (this_gulp_time - (this_gulp_time % acc_len) + acc_len)
                        else:
                            start_time = self.command_vals['start_time']
                        start = False
                        self.log.info("CORR >> New start time set to %d. Accumulating %d samples" % (start_time, acc_len))
                        ohdr['acc_len'] = acc_len
                        ohdr['seq0'] = start_time
                    self.update_stats({'curr_sample': this_gulp_time})
                    # If this is the start time, update the first flag, and compute where the last flag should be
                    if this_gulp_time == start_time:
                        self.log.info("CORR >> Start time %d reached." % start_time)
                        start = True
                        first = start_time
                        last  = first + acc_len - self.ntime_gulp
                        # on a new accumulation start, if a current oseq is open, close it, and start afresh
                        if oseq: oseq.end()
                        ohdr_str = json.dumps(ohdr)
                        self.sequence_proclog.update(ohdr)
                        oseq = oring.begin_sequence(time_tag=time_tag, header=ohdr_str, nringlet=iseq.nringlet)
                        time_tag += 1
                    if not start:
                        self.update_stats({'state': 'waiting'})
                        this_gulp_time += self.ntime_gulp
                        continue
                    self.update_stats({'state': 'running'})
                    # Use acc_len = 0 as a special stop condition
                    if acc_len == 0:
                        self.update_stats({'state': 'stopped'})
                        if oseq: oseq.end()
                        oseq = None
                        start = False

                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == first:
                        # reserve an output span
                        ospan = WriteSpan(oseq.ring, self.ogulp_size, nonblocking=False)
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                    if not ospan:
                        self.log.error("CORR: trying to write to not-yet-opened ospan")
                    idata = ispan.data_view('ci4').reshape(self.ntime_gulp,self.nchan,self.nstand*self.npol)
                    odata = ospan.data_view('ci32').reshape(self.nchan,self.nstand*(self.nstand+1)//2*self.npol*self.npol)
                    self.bfcc.execute(idata, odata, int(this_gulp_time==last))
                    curr_time = time.time()
                    process_time += curr_time - prev_time
                    prev_time = curr_time
                    if this_gulp_time == last:
                        ospan.close()
                        throughput_gbps = 8 * acc_len * ihdr['nchan'] * ihdr['nstand'] * ihdr['npol'] / process_time / 1e9
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,
                                                  'gbps': throughput_gbps})
                        self.update_stats({'last_end_sample': this_gulp_time, 'throughput': throughput_gbps})
                        process_time = 0
                        # Update integration boundary markers
                        first = last + self.ntime_gulp
                        last = first + acc_len - self.ntime_gulp
                    # And, update overall time counter
                    this_gulp_time += self.ntime_gulp
                if oseq: oseq.end()
                oseq = None
                start = False
                            
            # If upstream process stops producing, close things gracefully
            # TODO: why is this necessary? Get exceptions from ospan.__exit__ if not here
            if oseq:
                ospan.close()
                oseq.end()
