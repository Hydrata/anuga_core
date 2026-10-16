"""
Collect max stage quantity info
"""

__author__ = "gareth"
__date__ = "$01/08/2014 4:46:39 PM$"


import numpy as num
from anuga.operators.base_operator import Operator
from anuga import Quantity
from anuga.parallel import myid
from anuga.config import max_float


class Collect_max_quantities_operator(Operator):
    """
    Operator to collect the max stage, depth, speed, and momentum magnitude
    (speed * depth, i.e. ||(uh, vh)||) during a run.

    Maxima are updated every update_frequency timesteps [any integer >= 1],
    after t exceeds collection_start_time.

    If store_to_sww=True the running maxima are written to the SWW file every
    yield step as centroid quantities max_stage_c, max_depth_c, max_speed_c,
    max_uh_c.  This avoids the need to compute maxima by scanning all timeslices
    of the SWW stage output after the run.

    Velocity is zeroed below velocity_zero_height (defaults to
    minimum_allowed_height) to suppress spurious spikes in nearly-dry cells.

    When the domain is in GPU mode (MULTIPROCESSOR_GPU) the kernel runs
    entirely on the device every update; a device->host transfer of the four
    max arrays only happens in sync_to_host(), which the domain calls when it
    stores an SWW frame or writes a checkpoint (and export_max_quantities_to_csv()
    calls it too).  The host arrays self.max_* are therefore current only after
    sync_to_host().  On first GPU use the device arrays are seeded from the host
    arrays, so maxima restored from a checkpoint carry over into the resumed run.
    """

    def __init__(self,
                 domain,
                 update_frequency=1,
                 collection_start_time=0.,
                 velocity_zero_height=None,
                 store_to_sww=False,
                 description=None,
                 label=None,
                 logging=False,
                 verbose=False):

        Operator.__init__(self, domain, description, label, logging, verbose)

        self.domain = domain

        n = len(domain.centroid_coordinates[:, 0])
        self.max_stage = num.full(n, -max_float)
        self.max_depth = num.zeros(n)
        self.max_speed = num.zeros(n)
        self.max_uh    = num.zeros(n)

        self.xy    = domain.centroid_coordinates
        self.stage = domain.quantities['stage']
        self.elev  = domain.quantities['elevation']
        self.xmom  = domain.quantities['xmomentum']
        self.ymom  = domain.quantities['ymomentum']

        self.counter = 0
        assert update_frequency > 0, 'Update frequency must be >= 1'
        self.update_frequency = update_frequency
        self.collection_start_time = collection_start_time

        if velocity_zero_height is not None:
            self.velocity_zero_height = velocity_zero_height
        else:
            self.velocity_zero_height = domain.minimum_allowed_height

        self.store_to_sww = store_to_sww
        if store_to_sww:
            for qname in ('max_stage', 'max_depth', 'max_speed', 'max_uh'):
                Quantity(domain, name=qname, register=True, qty_type='centroid_only')
            # flag=4: centroid-only static variable overwritten each yield step
            domain.quantities_to_be_stored.update({
                'max_stage': 4,
                'max_depth': 4,
                'max_speed': 4,
                'max_uh':    4,
            })

        self._gpu_initialized = False
        self._gpu_dom = None
        # True once a max update has run; until then there is nothing to sync
        # (keeps the pre-update -max_float sentinel out of the stored frames).
        self._collected = False

    # ------------------------------------------------------------------
    # GPU initialisation (lazy, called on first __call__ in GPU mode)
    # ------------------------------------------------------------------

    def _init_gpu(self):
        if self._gpu_initialized and self._gpu_dom_or_none() is not None:
            return
        self._gpu_initialized = False

        domain = self.domain
        try:
            from anuga.config import MULTIPROCESSOR_GPU
        except ImportError:
            return

        if not hasattr(domain, 'multiprocessor_mode') or domain.multiprocessor_mode != MULTIPROCESSOR_GPU:
            return
        if not hasattr(domain, 'gpu_interface') or domain.gpu_interface is None:
            return
        gpu_interface = domain.gpu_interface
        if not hasattr(gpu_interface, 'gpu_dom') or gpu_interface.gpu_dom is None:
            return

        from anuga.shallow_water.sw_domain_gpu_ext import init_max_quantities_gpu
        n   = len(domain.centroid_coordinates[:, 0])
        # Seed the device arrays from the host maxima: fresh -inf/0 on a new
        # run, the restored peaks on a checkpoint resume.
        ret = init_max_quantities_gpu(gpu_interface.gpu_dom, n, self.velocity_zero_height,
                                      self.max_stage, self.max_depth,
                                      self.max_speed, self.max_uh)
        if ret != 0:
            raise RuntimeError(
                "gpu_max_quantities_init failed — check device memory availability"
            )
        self._gpu_initialized = True
        self._gpu_dom = gpu_interface.gpu_dom

    def _gpu_dom_or_none(self):
        """The GPUDomain whose device arrays hold the live maxima, else None."""
        if not self._gpu_initialized:
            return None
        gpu_interface = getattr(self.domain, 'gpu_interface', None)
        gpu_dom = getattr(gpu_interface, 'gpu_dom', None)
        if gpu_dom is None or gpu_dom is not self._gpu_dom:
            return None
        return gpu_dom

    def __getstate__(self):
        # The device arrays belong to a GPUDomain that is not pickled (the
        # domain drops gpu_interface).  After unpickling, re-initialise on the
        # new GPUDomain, seeded from the host arrays saved here.
        state = self.__dict__.copy()
        state['_gpu_initialized'] = False
        state['_gpu_dom'] = None
        return state

    def sync_to_host(self):
        """Bring the host-side maxima up to date.

        In GPU mode, copy the four device max arrays into self.max_stage,
        self.max_depth, self.max_speed and self.max_uh (one device->host
        transfer).  If store_to_sww=True, also copy them into the domain's
        max_* quantities for the SWW writer.  On the CPU path the host arrays
        are already current and this only refreshes the domain quantities.

        The domain calls this before storing an SWW frame and before writing a
        checkpoint, so the GPU kernel never pays a per-timestep transfer.
        """
        if not getattr(self, '_collected', True):
            return
        gpu_dom = self._gpu_dom_or_none()
        if gpu_dom is not None:
            from anuga.shallow_water.sw_domain_gpu_ext import get_max_quantities_gpu
            get_max_quantities_gpu(gpu_dom,
                                   self.max_stage, self.max_depth,
                                   self.max_speed, self.max_uh)
        if self.store_to_sww:
            self._copy_to_domain_quantities()

    def _copy_to_domain_quantities(self):
        d = self.domain
        d.quantities['max_stage'].centroid_values[:] = self.max_stage
        d.quantities['max_depth'].centroid_values[:] = self.max_depth
        d.quantities['max_speed'].centroid_values[:] = self.max_speed
        d.quantities['max_uh'].centroid_values[:]    = self.max_uh

    # ------------------------------------------------------------------

    def __call__(self):
        """
        Update max quantities every update_frequency timesteps once
        t > collection_start_time.
        """

        t = self.domain.get_time()

        if t >= self.collection_start_time:
            self.counter += 1

            if self.counter == self.update_frequency:
                self.counter = 0
                self._collected = True

                # ---- GPU path ----------------------------------------
                domain = self.domain
                try:
                    from anuga.config import MULTIPROCESSOR_GPU
                    _gpu_mode = (
                        hasattr(domain, 'multiprocessor_mode') and
                        domain.multiprocessor_mode == MULTIPROCESSOR_GPU and
                        hasattr(domain, 'gpu_interface') and
                        domain.gpu_interface is not None
                    )
                except ImportError:
                    _gpu_mode = False

                if _gpu_mode:
                    self._init_gpu()
                    if self._gpu_initialized:
                        from anuga.shallow_water.sw_domain_gpu_ext import \
                            update_max_quantities_gpu

                        # Device-only update; host arrays are refreshed by
                        # sync_to_host() when a frame is stored.
                        update_max_quantities_gpu(domain.gpu_interface.gpu_dom)
                        return

                # ---- CPU / NumPy path --------------------------------
                stage_c = self.stage.centroid_values
                elev_c  = self.elev.centroid_values
                xmom_c  = self.xmom.centroid_values
                ymom_c  = self.ymom.centroid_values

                self.max_stage = num.maximum(self.max_stage, stage_c)

                mom_norm = num.hypot(xmom_c, ymom_c)
                self.max_uh = num.maximum(self.max_uh, mom_norm)

                depth = num.maximum(stage_c - elev_c, 0.0)
                self.max_depth = num.maximum(self.max_depth, depth)

                vel = (mom_norm / num.maximum(depth, self.velocity_zero_height)) \
                      * (depth > self.velocity_zero_height)
                self.max_speed = num.maximum(self.max_speed, vel)

                if self.store_to_sww:
                    self._copy_to_domain_quantities()

    def parallel_safe(self):
        """Operator is applied independently on each cell and so is parallel safe."""
        return True

    def statistics(self):
        return self.label + ': Collect_max_quantity operator'

    def timestepping_statistics(self):
        from anuga import indent
        return indent + self.label + ': Collecting_max_quantity'

    def export_max_quantities_to_csv(self, filename_start='Max_Quantities_'):
        """Export max quantities to a CSV file."""

        # For GPU mode, ensure host arrays are up to date before export.
        domain = self.domain
        self.sync_to_host()

        full_ids = domain.tri_full_flag.nonzero()[0]
        geo = domain.geo_reference
        out = num.vstack([
            self.xy[full_ids, 0] + geo.xllcorner,
            self.xy[full_ids, 1] + geo.yllcorner,
            self.max_stage[full_ids],
            self.max_depth[full_ids],
            self.max_speed[full_ids],
            self.max_uh[full_ids],
        ]).T
        outname = filename_start + 'P' + str(myid) + '_X_Y_Stage_Depth_Speed_UH_MAX.csv'
        num.savetxt(outname, out, delimiter=',')
