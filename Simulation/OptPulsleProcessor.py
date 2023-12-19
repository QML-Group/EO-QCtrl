from collections.abc import Iterable
import warnings
import numbers

import numpy as np

from qutip import Qobj, identity, tensor, mesolve
import qutip_qtrl.pulseoptim as cpo
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device.processor import Processor
from qutip_qip.operations import gate_sequence_product, expand_operator
import qutip_qtrl.stats as stats
import qutip_qtrl.errors as errors
import qutip_qtrl.fidcomp as fidcomp
import qutip_qtrl.io as qtrlio
import timeit 
import scipy.optimize as spopt
import qutip.logging_utils as logging


logger = logging.get_logger()



__all__ = ["OptPulseProcessor"]

class OptimConfig(object):
    """
    Configuration parameters for control pulse optimisation

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip_qtrl.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.log_level = logger.getEffectiveLevel()
        self.alg = "GRAPE"  # Alts: 'CRAB'
        self.optim_method = "DEF"
        self.dyn_type = "DEF"
        self.fid_type = "DEF"
        self.fid_type = "DEF"
        self.tslot_type = "DEF"
        self.init_pulse_type = "DEF"

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    def check_create_output_dir(self, output_dir, desc="output"):
        """
        Checks if the given directory exists, if not it is created.

        Returns
        -------
        dir_ok : boolean
            True if directory exists (previously or created)
            False if failed to create the directory

        output_dir : string
            Path to the directory, which may be been made absolute

        msg : string
            Error msg if directory creation failed
        """
        return qtrlio.create_dir(output_dir, desc=desc)


# create global instance
optimconfig = OptimConfig()

warnings.simplefilter('always', DeprecationWarning) #turn off filter 
def _param_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

def _upper_safe(s):
    try:
        s = s.upper()
    except:
        pass
    return s

def _check_ctrls_container(ctrls):
    """
    Check through the controls container.
    Convert to an array if its a list of lists
    return the processed container
    raise type error if the container structure is invalid
    """
    if isinstance(ctrls, (list, tuple)):
        # Check to see if list of lists
        try:
            if isinstance(ctrls[0], (list, tuple)):
                ctrls_ = np.empty((len(ctrls), len(ctrls[0])), dtype=object)
                for i, ctrl in enumerate(ctrls):
                    ctrls_[i, :] = ctrl
                ctrls = ctrls_
        except IndexError:
            pass

    if isinstance(ctrls, np.ndarray):
        if len(ctrls.shape) != 2:
            raise TypeError("Incorrect shape for ctrl dyn gen array")
        for k in range(ctrls.shape[0]):
            for j in range(ctrls.shape[1]):
                if not isinstance(ctrls[k, j], Qobj):
                    raise TypeError("All control dyn gen must be Qobj")
    elif isinstance(ctrls, (list, tuple)):
        for ctrl in ctrls:
            if not isinstance(ctrl, Qobj):
                raise TypeError("All control dyn gen must be Qobj")
    else:
        raise TypeError("Controls list or array not set correctly")

    return ctrls


def _check_drift_dyn_gen(drift):
    if isinstance(drift, Qobj):
        return
    if not isinstance(drift, (list, tuple)):
        raise TypeError("drift should be a Qobj or a list of Qobj")
    for d in drift:
        if not isinstance(d, Qobj):
            raise TypeError("drift should be a Qobj or a list of Qobj")


def _attrib_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


def _func_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


class Dynamics(object):
    """
    This is a base class only. See subclass descriptions and choose an
    appropriate one for the application.

    Note that initialize_controls must be called before most of the methods
    can be used. init_timeslots can be called sometimes earlier in order
    to access timeslot related attributes

    This acts as a container for the operators that are used to calculate
    time evolution of the system under study. That is the dynamics generators
    (Hamiltonians, Lindbladians etc), the propagators from one timeslot to
    the next, and the evolution operators. Due to the large number of matrix
    additions and multiplications, for small systems at least, the optimisation
    performance is much better using ndarrays to represent these operators.
    However

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip_qtrl.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    params:  Dictionary
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.

    stats : Stats
        Attributes of which give performance stats for the optimisation
        set to None to reduce overhead of calculating stats.
        Note it is (usually) shared with the Optimizer object

    tslot_computer : TimeslotComputer (subclass instance)
        Used to manage when the timeslot dynamics
        generators, propagators, gradients etc are updated

    prop_computer : PropagatorComputer (subclass instance)
        Used to compute the propagators and their gradients

    fid_computer : FidelityComputer (subclass instance)
        Used to computer the fidelity error and the fidelity error
        gradient.

    memory_optimization : int
        Level of memory optimisation. Setting to 0 (default) means that
        execution speed is prioritized over memory.
        Setting to 1 means that some memory prioritisation steps will be
        taken, for instance using Qobj (and hence sparse arrays) as the
        the internal operator data type, and not caching some operators
        Potentially further memory saving maybe made with
        memory_optimization > 1.
        The options are processed in _set_memory_optimizations, see
        this for more information. Individual memory saving  options can be
        switched by settting them directly (see below)

    oper_dtype : type
        Data type for internal dynamics generators, propagators and time
        evolution operators. This can be ndarray or Qobj.
        Qobj may perform better for larger systems, and will also
        perform better when (custom) fidelity measures use Qobj methods
        such as partial trace.
        See _choose_oper_dtype for how this is chosen when not specified

    cache_phased_dyn_gen : bool
        If True then the dynamics generators will be saved with and
        without the propagation prefactor (if there is one)
        Defaults to True when memory_optimization=0, otherwise False

    cache_prop_grad : bool
        If the True then the propagator gradients (for exact gradients) will
        be computed when the propagator are computed and cache until
        the are used by the fidelity computer. If False then the
        fidelity computer will calculate them as needed.
        Defaults to True when memory_optimization=0, otherwise False

    cache_dyn_gen_eigenvectors_adj: bool
        If True then DynamicsUnitary will cached the adjoint of
        the Hamiltion eignvector matrix
        Defaults to True when memory_optimization=0, otherwise False

    sparse_eigen_decomp: bool
        If True then DynamicsUnitary will use the sparse eigenvalue
        decomposition.
        Defaults to True when memory_optimization<=1, otherwise False

    num_tslots : integer
        Number of timeslots (aka timeslices)

    num_ctrls : integer
        Number of controls.
        Note this is calculated as the length of ctrl_dyn_gen when first used.
        And is recalculated during initialise_controls only.

    evo_time : float
        Total time for the evolution

    tau : array[num_tslots] of float
        Duration of each timeslot
        Note that if this is set before initialize_controls is called
        then num_tslots and evo_time are calculated from tau, otherwise
        tau is generated from num_tslots and evo_time, that is
        equal size time slices

    time : array[num_tslots+1] of float
        Cumulative time for the evolution, that is the time at the start
        of each time slice

    drift_dyn_gen : Qobj or list of Qobj
        Drift or system dynamics generator (Hamiltonian)
        Matrix defining the underlying dynamics of the system
        Can also be a list of Qobj (length num_tslots) for time varying
        drift dynamics

    ctrl_dyn_gen : List of Qobj
        Control dynamics generator (Hamiltonians)
        List of matrices defining the control dynamics

    initial : Qobj
        Starting state / gate
        The matrix giving the initial state / gate, i.e. at time 0
        Typically the identity for gate evolution

    target : Qobj
        Target state / gate:
        The matrix giving the desired state / gate for the evolution

    ctrl_amps : array[num_tslots, num_ctrls] of float
        Control amplitudes
        The amplitude (scale factor) for each control in each timeslot

    initial_ctrl_scaling : float
        Scale factor applied to be applied the control amplitudes
        when they are initialised
        This is used by the PulseGens rather than in any fucntions in
        this class

    initial_ctrl_offset  : float
        Linear offset applied to be applied the control amplitudes
        when they are initialised
        This is used by the PulseGens rather than in any fucntions in
        this class

    dyn_gen : List of Qobj
        Dynamics generators
        the combined drift and control dynamics generators
        for each timeslot

    prop : list of Qobj
        Propagators - used to calculate time evolution from one
        timeslot to the next

    prop_grad : array[num_tslots, num_ctrls] of Qobj
        Propagator gradient (exact gradients only)
        Array  of Qobj that give the gradient
        with respect to the control amplitudes in a timeslot
        Note this attribute is only created when the selected
        PropagatorComputer is an exact gradient type.

    fwd_evo : List of Qobj
        Forward evolution (or propagation)
        the time evolution operator from the initial state / gate to the
        specified timeslot as generated by the dyn_gen

    onwd_evo : List of Qobj
        Onward evolution (or propagation)
        the time evolution operator from the specified timeslot to
        end of the evolution time as generated by the dyn_gen

    onto_evo : List of Qobj
        'Backward' List of Qobj propagation
        the overlap of the onward propagation with the inverse of the
        target.
        Note this is only used (so far) by the unitary dynamics fidelity

    evo_current : Boolean
        Used to flag that the dynamics used to calculate the evolution
        operators is current. It is set to False when the amplitudes
        change

    fact_mat_round_prec : float
        Rounding precision used when calculating the factor matrix
        to determine if two eigenvalues are equivalent
        Only used when the PropagatorComputer uses diagonalisation

    def_amps_fname : string
        Default name for the output used when save_amps is called

    unitarity_check_level : int
        If > 0 then unitarity of the system evolution is checked at at
        evolution recomputation.
        level 1 checks all propagators
        level 2 checks eigen basis as well
        Default is 0

    unitarity_tol :
        Tolerance used in checking if operator is unitary
        Default is 1e-10

    dump : :class:`qutip.control.dump.DynamicsDump`
        Store of historical calculation data.
        Set to None (Default) for no storing of historical data
        Use dumping property to set level of data dumping

    dumping : string
        level of data dumping: NONE, SUMMARY, FULL or CUSTOM
        See property docstring for details

    dump_to_file : bool
        If set True then data will be dumped to file during the calculations
        dumping will be set to SUMMARY during init_evo if dump_to_file is True
        and dumping not set.
        Default is False

    dump_dir : string
        Basically a link to dump.dump_dir. Exists so that it can be set through
        dyn_params.
        If dump is None then will return None or will set dumping to SUMMARY
        when setting a path

    """

    def __init__(self, optimconfig, params=None):
        self.config = optimconfig
        self.params = params
        self.reset()

    def reset(self):
        # Link to optimiser object if self is linked to one
        self.parent = None
        # Main functional attributes
        self.time = None
        self.initial = None
        self.target = None
        self.ctrl_amps = None
        self.initial_ctrl_scaling = 1.0
        self.initial_ctrl_offset = 0.0
        self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self._tau = None
        self._evo_time = None
        self._num_ctrls = None
        self._num_tslots = None
        # attributes used for processing evolution
        self.memory_optimization = 0
        self.oper_dtype = None
        self.cache_phased_dyn_gen = None
        self.cache_prop_grad = None
        self.cache_dyn_gen_eigenvectors_adj = None
        self.sparse_eigen_decomp = None
        self.dyn_dims = None
        self.dyn_shape = None
        self.sys_dims = None
        self.sys_shape = None
        self.time_depend_drift = False
        self.time_depend_ctrl_dyn_gen = False
        # These internal attributes will be of the internal operator data type
        # used to compute the evolution
        # This will be either ndarray or Qobj
        self._drift_dyn_gen = None
        self._ctrl_dyn_gen = None
        self._phased_ctrl_dyn_gen = None
        self._dyn_gen_phase = None
        self._phase_application = None
        self._initial = None
        self._target = None
        self._onto_evo_target = None
        self._dyn_gen = None
        self._phased_dyn_gen = None
        self._prop = None
        self._prop_grad = None
        self._fwd_evo = None
        self._onwd_evo = None
        self._onto_evo = None
        # The _qobj attribs are Qobj representations of the equivalent
        # internal attribute. They are only set when the extenal accessors
        # are used
        self._onto_evo_target_qobj = None
        self._dyn_gen_qobj = None
        self._prop_qobj = None
        self._prop_grad_qobj = None
        self._fwd_evo_qobj = None
        self._onwd_evo_qobj = None
        self._onto_evo_qobj = None
        # Atrributes used in diagonalisation
        # again in internal operator data type (see above)
        self._decomp_curr = None
        self._prop_eigen = None
        self._dyn_gen_eigenvectors = None
        self._dyn_gen_eigenvectors_adj = None
        self._dyn_gen_factormatrix = None
        self.fact_mat_round_prec = 1e-10

        # Debug and information attribs
        self.stats = None
        self.id_text = "DYN_BASE"
        self.def_amps_fname = "ctrl_amps.txt"
        self.log_level = self.config.log_level
        # Internal flags
        self._dyn_gen_mapped = False
        self._evo_initialized = False
        self._timeslots_initialized = False
        self._ctrls_initialized = False
        self._ctrl_dyn_gen_checked = False
        self._drift_dyn_gen_checked = False
        # Unitary checking
        self.unitarity_check_level = 0
        self.unitarity_tol = 1e-10
        # Data dumping
        self.dump = None
        self.dump_to_file = False

        self.apply_params()

        # Create the computing objects
        self._create_computers()

        self.clear()

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    @property
    def dumping(self):
        """
        The level of data dumping that will occur during the time evolution
        calculation.

        - NONE : No processing data dumped (Default)
        - SUMMARY : A summary of each time evolution will be recorded
        - FULL : All operators used or created in the calculation dumped
        - CUSTOM : Some customised level of dumping

        When first set to CUSTOM this is equivalent to SUMMARY. It is then up
        to the user to specify which operators are dumped.  WARNING: FULL could
        consume a lot of memory!
        """
        if self.dump is None:
            lvl = "NONE"
        else:
            lvl = self.dump.level

        return lvl

    @dumping.setter
    def dumping(self, value):
        if value is None:
            self.dump = None
        else:
            if not isinstance(value, str):
                raise TypeError("Value must be string value")
            lvl = value.upper()
            if lvl == "NONE":
                self.dump = None
            else:
                if not isinstance(self.dump, qtrldump.DynamicsDump):
                    self.dump = qtrldump.DynamicsDump(self, level=lvl)
                else:
                    self.dump.level = lvl

    @property
    def dump_dir(self):
        if self.dump:
            return self.dump.dump_dir
        else:
            return None

    @dump_dir.setter
    def dump_dir(self, value):
        if not self.dump:
            self.dumping = "SUMMARY"
        self.dump.dump_dir = value

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to UpdateAll
        # can be set to DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == "DYNAMIC":
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    def clear(self):
        self.ctrl_amps = None
        self.evo_current = False
        if self.fid_computer is not None:
            self.fid_computer.clear()

    @property
    def num_tslots(self):
        if not self._timeslots_initialized:
            self.init_timeslots()
        return self._num_tslots

    @num_tslots.setter
    def num_tslots(self, value):
        self._num_tslots = value
        if self._timeslots_initialized:
            self._tau = None
            self.init_timeslots()

    @property
    def evo_time(self):
        if not self._timeslots_initialized:
            self.init_timeslots()
        return self._evo_time

    @evo_time.setter
    def evo_time(self, value):
        self._evo_time = value
        if self._timeslots_initialized:
            self._tau = None
            self.init_timeslots()

    @property
    def tau(self):
        if not self._timeslots_initialized:
            self.init_timeslots()
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value
        self.init_timeslots()

    def init_timeslots(self):
        """
        Generate the timeslot duration array 'tau' based on the evo_time
        and num_tslots attributes, unless the tau attribute is already set
        in which case this step in ignored
        Generate the cumulative time array 'time' based on the tau values
        """
        # set the time intervals to be equal timeslices of the total if
        # the have not been set already (as part of user config)
        if self._num_tslots is None:
            self._num_tslots = DEF_NUM_TSLOTS
        if self._evo_time is None:
            self._evo_time = DEF_EVO_TIME

        if self._tau is None:
            self._tau = (
                np.ones(self._num_tslots, dtype="f")
                * self._evo_time
                / self._num_tslots
            )
        else:
            self._num_tslots = len(self._tau)
            self._evo_time = np.sum(self._tau)

        self.time = np.zeros(self._num_tslots + 1, dtype=float)
        # set the cumulative time by summing the time intervals
        for t in range(self._num_tslots):
            self.time[t + 1] = self.time[t] + self._tau[t]

        self._timeslots_initialized = True

    def _set_memory_optimizations(self):
        """
        Set various memory optimisation attributes based on the
        memory_optimization attribute
        If they have been set already, e.g. in apply_params
        then they will not be overridden here
        """
        logger.info(
            "Setting memory optimisations for level {}".format(
                self.memory_optimization
            )
        )

        if self.oper_dtype is None:
            self._choose_oper_dtype()
            logger.info(
                "Internal operator data type choosen to be {}".format(
                    self.oper_dtype
                )
            )
        else:
            logger.info("Using operator data type {}".format(self.oper_dtype))

        if self.cache_phased_dyn_gen is None:
            if self.memory_optimization > 0:
                self.cache_phased_dyn_gen = False
            else:
                self.cache_phased_dyn_gen = True
        logger.info(
            "phased dynamics generator caching {}".format(
                self.cache_phased_dyn_gen
            )
        )

        if self.cache_prop_grad is None:
            if self.memory_optimization > 0:
                self.cache_prop_grad = False
            else:
                self.cache_prop_grad = True
        logger.info(
            "propagator gradient caching {}".format(self.cache_prop_grad)
        )

        if self.cache_dyn_gen_eigenvectors_adj is None:
            if self.memory_optimization > 0:
                self.cache_dyn_gen_eigenvectors_adj = False
            else:
                self.cache_dyn_gen_eigenvectors_adj = True
        logger.info(
            "eigenvector adjoint caching {}".format(
                self.cache_dyn_gen_eigenvectors_adj
            )
        )

        if self.sparse_eigen_decomp is None:
            if self.memory_optimization > 1:
                self.sparse_eigen_decomp = True
            else:
                self.sparse_eigen_decomp = False
        logger.info(
            "use sparse eigen decomp {}".format(self.sparse_eigen_decomp)
        )

    def _choose_oper_dtype(self):
        """
        Attempt select most efficient internal operator data type
        """

        if self.memory_optimization > 0:
            self.oper_dtype = Qobj
        else:
            # Method taken from Qobj.expm()
            # if method is not explicitly given, try to make a good choice
            # between sparse and dense solvers by considering the size of the
            # system and the number of non-zero elements.
            if self.time_depend_drift:
                dg = self.drift_dyn_gen[0]
            else:
                dg = self.drift_dyn_gen
            if self.time_depend_ctrl_dyn_gen:
                ctrls = self.ctrl_dyn_gen[0, :]
            else:
                ctrls = self.ctrl_dyn_gen
            for c in ctrls:
                dg = dg + c

            N = dg.shape[0]
            if isinstance(dg.data, _data.CSR):
                n = _data.csr.nnz(dg.data)
            else:
                n = N**2

            if N**2 < 100 * n:
                # large number of nonzero elements, revert to dense solver
                self.oper_dtype = np.ndarray
            elif N > 400:
                # large system, and quite sparse -> qutips sparse method
                self.oper_dtype = Qobj
            else:
                # small system, but quite sparse -> qutips sparse/dense method
                self.oper_dtype = np.ndarray

        return self.oper_dtype

    def _init_evo(self):
        """
        Create the container lists / arrays for the:
        dynamics generations, propagators, and evolutions etc
        Set the time slices and cumulative time
        """
        # check evolution operators
        if not self._drift_dyn_gen_checked:
            _check_drift_dyn_gen(self.drift_dyn_gen)
        if not self._ctrl_dyn_gen_checked:
            self.ctrl_dyn_gen = _check_ctrls_container(self.ctrl_dyn_gen)

        if not isinstance(self.initial, Qobj):
            raise TypeError("initial must be a Qobj")

        if not isinstance(self.target, Qobj):
            raise TypeError("target must be a Qobj")

        self.refresh_drift_attribs()
        self.sys_dims = self.initial.dims
        self.sys_shape = self.initial.shape
        # Set the phase application method
        self._init_phase()

        self._set_memory_optimizations()
        if self.sparse_eigen_decomp and self.sys_shape[0] <= 2:
            raise ValueError(
                "Single qubit pulse optimization dynamics cannot use sparse"
                " eigenvector decomposition because of limitations in"
                " scipy.linalg.eigsh. Pleae set sparse_eigen_decomp to False"
                " or increase the size of the system."
            )

        n_ts = self.num_tslots
        n_ctrls = self.num_ctrls
        if self.oper_dtype == Qobj:
            self._initial = self.initial
            self._target = self.target
            self._drift_dyn_gen = self.drift_dyn_gen
            self._ctrl_dyn_gen = self.ctrl_dyn_gen
        elif self.oper_dtype == np.ndarray:
            self._initial = self.initial.full()
            self._target = self.target.full()
            if self.time_depend_drift:
                self._drift_dyn_gen = [d.full() for d in self.drift_dyn_gen]
            else:
                self._drift_dyn_gen = self.drift_dyn_gen.full()
            if self.time_depend_ctrl_dyn_gen:
                self._ctrl_dyn_gen = np.empty([n_ts, n_ctrls], dtype=object)
                for k in range(n_ts):
                    for j in range(n_ctrls):
                        self._ctrl_dyn_gen[k, j] = self.ctrl_dyn_gen[
                            k, j
                        ].full()
            else:
                self._ctrl_dyn_gen = [
                    ctrl.full() for ctrl in self.ctrl_dyn_gen
                ]
        else:
            raise ValueError(
                "Unknown oper_dtype {!r}. The oper_dtype may be qutip.Qobj or"
                " numpy.ndarray.".format(self.oper_dtype)
            )

        if self.cache_phased_dyn_gen:
            if self.time_depend_ctrl_dyn_gen:
                self._phased_ctrl_dyn_gen = np.empty(
                    [n_ts, n_ctrls], dtype=object
                )
                for k in range(n_ts):
                    for j in range(n_ctrls):
                        self._phased_ctrl_dyn_gen[k, j] = self._apply_phase(
                            self._ctrl_dyn_gen[k, j]
                        )
            else:
                self._phased_ctrl_dyn_gen = [
                    self._apply_phase(ctrl) for ctrl in self._ctrl_dyn_gen
                ]

        self._dyn_gen = [object for x in range(self.num_tslots)]
        if self.cache_phased_dyn_gen:
            self._phased_dyn_gen = [object for x in range(self.num_tslots)]
        self._prop = [object for x in range(self.num_tslots)]
        if self.prop_computer.grad_exact and self.cache_prop_grad:
            self._prop_grad = np.empty(
                [self.num_tslots, self.num_ctrls], dtype=object
            )
        # Time evolution operator (forward propagation)
        self._fwd_evo = [object for x in range(self.num_tslots + 1)]
        self._fwd_evo[0] = self._initial
        if self.fid_computer.uses_onwd_evo:
            # Time evolution operator (onward propagation)
            self._onwd_evo = [object for x in range(self.num_tslots)]
        if self.fid_computer.uses_onto_evo:
            # Onward propagation overlap with inverse target
            self._onto_evo = [object for x in range(self.num_tslots + 1)]
            self._onto_evo[self.num_tslots] = self._get_onto_evo_target()

        if isinstance(self.prop_computer, propcomp.PropCompDiag):
            self._create_decomp_lists()

        if self.log_level <= logging.DEBUG and isinstance(
            self, DynamicsUnitary
        ):
            self.unitarity_check_level = 1

        if self.dump_to_file:
            if self.dump is None:
                self.dumping = "SUMMARY"
            self.dump.write_to_file = True
            self.dump.create_dump_dir()
            logger.info(
                "Dynamics dump will be written to:\n{}".format(
                    self.dump.dump_dir
                )
            )

        self._evo_initialized = True

    @property
    def dyn_gen_phase(self):
        """
        Some op that is applied to the dyn_gen before expontiating to
        get the propagator.
        See `phase_application` for how this is applied
        """
        # Note that if this returns None then _apply_phase will never be
        # called
        return self._dyn_gen_phase

    @dyn_gen_phase.setter
    def dyn_gen_phase(self, value):
        self._dyn_gen_phase = value

    @property
    def phase_application(self):
        """
        phase_application : scalar(string), default='preop'
        Determines how the phase is applied to the dynamics generators

        - 'preop'  : P = expm(phase*dyn_gen)
        - 'postop' : P = expm(dyn_gen*phase)
        - 'custom' : Customised phase application

        The 'custom' option assumes that the _apply_phase method has been
        set to a custom function.
        """
        return self._phase_application

    @phase_application.setter
    def phase_application(self, value):
        self._set_phase_application(value)

    def _set_phase_application(self, value):
        self._config_phase_application(value)
        self._phase_application = value

    def _config_phase_application(self, ph_app=None):
        """
        Set the appropriate function for the phase application
        """
        err_msg = (
            "Invalid value '{}' for phase application. Must be either "
            "'preop', 'postop' or 'custom'".format(ph_app)
        )

        if ph_app is None:
            ph_app = self._phase_application

        try:
            ph_app = ph_app.lower()
        except AttributeError:
            raise ValueError(err_msg)

        if ph_app == "preop":
            self._apply_phase = self._apply_phase_preop
        elif ph_app == "postop":
            self._apply_phase = self._apply_phase_postop
        elif ph_app == "custom":
            # Do nothing, assume _apply_phase set elsewhere
            pass
        else:
            raise ValueError(err_msg)

    def _init_phase(self):
        if self.dyn_gen_phase is not None:
            self._config_phase_application()
        else:
            self.cache_phased_dyn_gen = False

    def _apply_phase(self, dg):
        """
        This default method does nothing.
        It will be set to another method automatically if `phase_application`
        is 'preop' or 'postop'. It should be overridden repointed if
        `phase_application` is 'custom'
        It will never be called if `dyn_gen_phase` is None
        """
        return dg

    def _apply_phase_preop(self, dg):
        """
        Apply phasing operator to dynamics generator.
        This called during the propagator calculation.
        In this case it will be applied as phase*dg
        """
        if hasattr(self.dyn_gen_phase, "dot"):
            phased_dg = self._dyn_gen_phase.dot(dg)
        else:
            phased_dg = self._dyn_gen_phase * dg
        return phased_dg

    def _apply_phase_postop(self, dg):
        """
        Apply phasing operator to dynamics generator.
        This called during the propagator calculation.
        In this case it will be applied as dg*phase
        """
        if hasattr(self.dyn_gen_phase, "dot"):
            phased_dg = dg.dot(self._dyn_gen_phase)
        else:
            phased_dg = dg * self._dyn_gen_phase
        return phased_dg

    def _create_decomp_lists(self):
        """
        Create lists that will hold the eigen decomposition
        used in calculating propagators and gradients
        Note: used with PropCompDiag propagator calcs
        """
        n_ts = self.num_tslots
        self._decomp_curr = [False for x in range(n_ts)]
        self._prop_eigen = [object for x in range(n_ts)]
        self._dyn_gen_eigenvectors = [object for x in range(n_ts)]
        if self.cache_dyn_gen_eigenvectors_adj:
            self._dyn_gen_eigenvectors_adj = [object for x in range(n_ts)]
        self._dyn_gen_factormatrix = [object for x in range(n_ts)]

    def initialize_controls(self, amps, init_tslots=True):
        """
        Set the initial control amplitudes and time slices
        Note this must be called after the configuration is complete
        before any dynamics can be calculated
        """
        if not isinstance(self.prop_computer, propcomp.PropagatorComputer):
            raise errors.UsageError(
                "No prop_computer (propagator computer) "
                "set. A default should be assigned by the Dynamics subclass"
            )

        if not isinstance(self.tslot_computer, tslotcomp.TimeslotComputer):
            raise errors.UsageError(
                "No tslot_computer (Timeslot computer)"
                " set. A default should be assigned by the Dynamics class"
            )

        if not isinstance(self.fid_computer, fidcomp.FidelityComputer):
            raise errors.UsageError(
                "No fid_computer (Fidelity computer)"
                " set. A default should be assigned by the Dynamics subclass"
            )

        self.ctrl_amps = None
        if not self._timeslots_initialized:
            init_tslots = True
        if init_tslots:
            self.init_timeslots()
        self._init_evo()
        self.tslot_computer.init_comp()
        self.fid_computer.init_comp()
        self._ctrls_initialized = True
        self.update_ctrl_amps(amps)

    def check_ctrls_initialized(self):
        if not self._ctrls_initialized:
            raise errors.UsageError(
                "Controls not initialised. "
                "Ensure Dynamics.initialize_controls has been "
                "executed with the initial control amplitudes."
            )

    def get_amp_times(self):
        return self.time[: self.num_tslots]

    def save_amps(self, file_name=None, times=None, amps=None, verbose=False):
        """
        Save a file with the current control amplitudes in each timeslot
        The first column in the file will be the start time of the slot

        Parameters
        ----------
        file_name : string
            Name of the file
            If None given the def_amps_fname attribuite will be used

        times : List type (or string)
            List / array of the start times for each slot
            If None given this will be retrieved through get_amp_times()
            If 'exclude' then times will not be saved in the file, just
            the amplitudes

        amps : Array[num_tslots, num_ctrls]
            Amplitudes to be saved
            If None given the ctrl_amps attribute will be used

        verbose : Boolean
            If True then an info message will be logged
        """
        self.check_ctrls_initialized()

        inctimes = True
        if file_name is None:
            file_name = self.def_amps_fname
        if amps is None:
            amps = self.ctrl_amps
        if times is None:
            times = self.get_amp_times()
        else:
            if isinstance(times, str):
                if times.lower() == "exclude":
                    inctimes = False
                else:
                    logger.warn(
                        "Unknown option for times '{}' "
                        "when saving amplitudes".format(times)
                    )
                    times = self.get_amp_times()

        try:
            if inctimes:
                shp = amps.shape
                data = np.empty([shp[0], shp[1] + 1], dtype=float)
                data[:, 0] = times
                data[:, 1:] = amps
            else:
                data = amps

            np.savetxt(file_name, data, delimiter="\t", fmt="%14.6g")

            if verbose:
                logger.info("Amplitudes saved to file: " + file_name)
        except Exception as e:
            logger.error(
                "Failed to save amplitudes due to underling "
                "error: {}".format(e)
            )

    def update_ctrl_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        The actual work is completed by the compare_amps method of the
        timeslot computer
        """

        if self.log_level <= logging.DEBUG_INTENSE:
            logger.log(
                logging.DEBUG_INTENSE,
                "Updating amplitudes...\n"
                "Current control amplitudes:\n"
                + str(self.ctrl_amps)
                + "\n(potenially) new amplitudes:\n"
                + str(new_amps),
            )

        self.tslot_computer.compare_amps(new_amps)

    def flag_system_changed(self):
        """
        Flag evolution, fidelity and gradients as needing recalculation
        """
        self.evo_current = False
        self.fid_computer.flag_system_changed()

    def get_drift_dim(self):
        """
        Returns the size of the matrix that defines the drift dynamics
        that is assuming the drift is NxN, then this returns N
        """
        if self.dyn_shape is None:
            self.refresh_drift_attribs()
        return self.dyn_shape[0]

    def refresh_drift_attribs(self):
        """Reset the dyn_shape, dyn_dims and time_depend_drift attribs"""

        if isinstance(self.drift_dyn_gen, (list, tuple)):
            d0 = self.drift_dyn_gen[0]
            self.time_depend_drift = True
        else:
            d0 = self.drift_dyn_gen
            self.time_depend_drift = False

        if not isinstance(d0, Qobj):
            raise TypeError(
                "Unable to determine drift attributes, "
                "because drift_dyn_gen is not Qobj (nor list of)"
            )

        self.dyn_shape = d0.shape
        self.dyn_dims = d0.dims

    def get_num_ctrls(self):
        """
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively
        subsequently
        """
        _func_deprecation(
            "'get_num_ctrls' has been replaced by " "'num_ctrls' property"
        )
        return self.num_ctrls

    def _get_num_ctrls(self):
        if not self._ctrl_dyn_gen_checked:
            self.ctrl_dyn_gen = _check_ctrls_container(self.ctrl_dyn_gen)
            self._ctrl_dyn_gen_checked = True
        if isinstance(self.ctrl_dyn_gen, np.ndarray):
            self._num_ctrls = self.ctrl_dyn_gen.shape[1]
            self.time_depend_ctrl_dyn_gen = True
        else:
            self._num_ctrls = len(self.ctrl_dyn_gen)

        return self._num_ctrls

    @property
    def num_ctrls(self):
        """
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively
        subsequently
        """
        if self._num_ctrls is None:
            self._num_ctrls = self._get_num_ctrls()
        return self._num_ctrls

    @property
    def onto_evo_target(self):
        if self._onto_evo_target is None:
            self._get_onto_evo_target()

        if self._onto_evo_target_qobj is None:
            if isinstance(self._onto_evo_target, Qobj):
                self._onto_evo_target_qobj = self._onto_evo_target
            else:
                rev_dims = [self.sys_dims[1], self.sys_dims[0]]
                self._onto_evo_target_qobj = Qobj(
                    self._onto_evo_target, dims=rev_dims
                )

        return self._onto_evo_target_qobj

    def get_owd_evo_target(self):
        _func_deprecation(
            "'get_owd_evo_target' has been replaced by "
            "'onto_evo_target' property"
        )
        return self.onto_evo_target

    def _get_onto_evo_target(self):
        """
        Get the inverse of the target.
        Used for calculating the 'onto target' evolution
        This is actually only relevant for unitary dynamics where
        the target.dag() is what is required
        However, for completeness, in general the inverse of the target
        operator is is required
        For state-to-state, the bra corresponding to the is required ket
        """
        if self.target.shape[0] == self.target.shape[1]:
            # Target is operator
            targ = la.inv(self.target.full())
            if self.oper_dtype == Qobj:
                rev_dims = [self.target.dims[1], self.target.dims[0]]
                self._onto_evo_target = Qobj(targ, dims=rev_dims)
            elif self.oper_dtype == np.ndarray:
                self._onto_evo_target = targ
            else:
                assert False, f"Unknown oper_dtype {self.oper_dtype!r}"
        else:
            if self.oper_dtype == Qobj:
                self._onto_evo_target = self.target.dag()
            elif self.oper_dtype == np.ndarray:
                self._onto_evo_target = self.target.dag().full()
            else:
                assert False, f"Unknown oper_dtype {self.oper_dtype!r}"

        return self._onto_evo_target

    def combine_dyn_gen(self, k):
        """
        Computes the dynamics generator for a given timeslot
        The is the combined Hamiltion for unitary systems
        """
        _func_deprecation(
            "'combine_dyn_gen' has been replaced by " "'_combine_dyn_gen'"
        )
        self._combine_dyn_gen(k)
        return self._dyn_gen(k)

    def _combine_dyn_gen(self, k):
        """
        Computes the dynamics generator for a given timeslot
        The is the combined Hamiltion for unitary systems
        Also applies the phase (if any required by the propagation)
        """
        if self.time_depend_drift:
            dg = self._drift_dyn_gen[k]
        else:
            dg = self._drift_dyn_gen
        for j in range(self._num_ctrls):
            if self.time_depend_ctrl_dyn_gen:
                dg = dg + self.ctrl_amps[k, j] * self._ctrl_dyn_gen[k, j]
            else:
                dg = dg + self.ctrl_amps[k, j] * self._ctrl_dyn_gen[j]

        self._dyn_gen[k] = dg
        if self.cache_phased_dyn_gen:
            self._phased_dyn_gen[k] = self._apply_phase(dg)

    def get_dyn_gen(self, k):
        """
        Get the combined dynamics generator for the timeslot
        Not implemented in the base class. Choose a subclass
        """
        _func_deprecation(
            "'get_dyn_gen' has been replaced by " "'_get_phased_dyn_gen'"
        )
        return self._get_phased_dyn_gen(k)

    def _get_phased_dyn_gen(self, k):
        if self.dyn_gen_phase is None:
            return self._dyn_gen[k]
        else:
            if self._phased_dyn_gen is None:
                return self._apply_phase(self._dyn_gen[k])
            else:
                return self._phased_dyn_gen[k]

    def get_ctrl_dyn_gen(self, j):
        """
        Get the dynamics generator for the control
        Not implemented in the base class. Choose a subclass
        """
        _func_deprecation(
            "'get_ctrl_dyn_gen' has been replaced by "
            "'_get_phased_ctrl_dyn_gen'"
        )
        return self._get_phased_ctrl_dyn_gen(0, j)

    def _get_phased_ctrl_dyn_gen(self, k, j):
        if self._phased_ctrl_dyn_gen is not None:
            if self.time_depend_ctrl_dyn_gen:
                return self._phased_ctrl_dyn_gen[k, j]
            else:
                return self._phased_ctrl_dyn_gen[j]
        else:
            if self.time_depend_ctrl_dyn_gen:
                if self.dyn_gen_phase is None:
                    return self._ctrl_dyn_gen[k, j]
                else:
                    return self._apply_phase(self._ctrl_dyn_gen[k, j])
            else:
                if self.dyn_gen_phase is None:
                    return self._ctrl_dyn_gen[j]
                else:
                    return self._apply_phase(self._ctrl_dyn_gen[j])

    @property
    def dyn_gen(self):
        """
        List of combined dynamics generators (Qobj) for each timeslot
        """
        if self._dyn_gen is not None:
            if self._dyn_gen_qobj is None:
                if self.oper_dtype == Qobj:
                    self._dyn_gen_qobj = self._dyn_gen
                else:
                    self._dyn_gen_qobj = [
                        Qobj(dg, dims=self.dyn_dims) for dg in self._dyn_gen
                    ]
        return self._dyn_gen_qobj

    @property
    def prop(self):
        """
        List of propagators (Qobj) for each timeslot
        """
        if self._prop is not None:
            if self._prop_qobj is None:
                if self.oper_dtype == Qobj:
                    self._prop_qobj = self._prop
                else:
                    self._prop_qobj = [
                        Qobj(dg, dims=self.dyn_dims) for dg in self._prop
                    ]
        return self._prop_qobj

    @property
    def prop_grad(self):
        """
        Array of propagator gradients (Qobj) for each timeslot, control
        """
        if self._prop_grad is not None:
            if self._prop_grad_qobj is None:
                if self.oper_dtype == Qobj:
                    self._prop_grad_qobj = self._prop_grad
                else:
                    self._prop_grad_qobj = np.empty(
                        [self.num_tslots, self.num_ctrls], dtype=object
                    )
                    for k in range(self.num_tslots):
                        for j in range(self.num_ctrls):
                            self._prop_grad_qobj[k, j] = Qobj(
                                self._prop_grad[k, j], dims=self.dyn_dims
                            )
        return self._prop_grad_qobj

    def _get_prop_grad(self, k, j):
        if self.cache_prop_grad:
            prop_grad = self._prop_grad[k, j]
        else:
            prop_grad = self.prop_computer._compute_prop_grad(
                k, j, compute_prop=False
            )
        return prop_grad

    @property
    def evo_init2t(self):
        _attrib_deprecation("'evo_init2t' has been replaced by '_fwd_evo'")
        return self._fwd_evo

    @property
    def fwd_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        if self._fwd_evo is not None:
            if self._fwd_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._fwd_evo_qobj = self._fwd_evo
                else:
                    self._fwd_evo_qobj = [self.initial]
                    for k in range(1, self.num_tslots + 1):
                        self._fwd_evo_qobj.append(
                            Qobj(self._fwd_evo[k], dims=self.sys_dims)
                        )
        return self._fwd_evo_qobj

    def _get_full_evo(self):
        return self._fwd_evo[self._num_tslots]

    @property
    def full_evo(self):
        """Full evolution - time evolution at final time slot"""
        return self.fwd_evo[self.num_tslots]

    @property
    def evo_t2end(self):
        _attrib_deprecation("'evo_t2end' has been replaced by '_onwd_evo'")
        return self._onwd_evo

    @property
    def onwd_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        if self._onwd_evo is not None:
            if self._onwd_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._onwd_evo_qobj = self._fwd_evo
                else:
                    self._onwd_evo_qobj = [
                        Qobj(dg, dims=self.sys_dims) for dg in self._onwd_evo
                    ]
        return self._onwd_evo_qobj

    @property
    def evo_t2targ(self):
        _attrib_deprecation("'evo_t2targ' has been replaced by '_onto_evo'")
        return self._onto_evo

    @property
    def onto_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        if self._onto_evo is not None:
            if self._onto_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._onto_evo_qobj = self._onto_evo
                else:
                    self._onto_evo_qobj = []
                    for k in range(0, self.num_tslots):
                        self._onto_evo_qobj.append(
                            Qobj(self._onto_evo[k], dims=self.sys_dims)
                        )
                    self._onto_evo_qobj.append(self.onto_evo_target)

        return self._onto_evo_qobj

    def compute_evolution(self):
        """
        Recalculate the time evolution operators
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary
        Actual work is completed by the recompute_evolution method
        of the timeslot computer
        """

        # Check if values are already current, otherwise calculate all values
        if not self.evo_current:
            if self.log_level <= logging.DEBUG_VERBOSE:
                logger.log(logging.DEBUG_VERBOSE, "Computing evolution")
            self.tslot_computer.recompute_evolution()
            self.evo_current = True
            return True
        return False

    def _ensure_decomp_curr(self, k):
        """
        Checks to see if the diagonalisation has been completed since
        the last update of the dynamics generators
        (after the amplitude update)
        If not then the diagonlisation is completed
        """
        if self._decomp_curr is None:
            raise errors.UsageError("Decomp lists have not been created")
        if not self._decomp_curr[k]:
            self._spectral_decomp(k)

    def _spectral_decomp(self, k):
        """
        Calculate the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient
        Not implemented in this base class, because the method is specific
        to the matrix type
        """
        raise errors.UsageError(
            "Decomposition cannot be completed by "
            "this class. Try a(nother) subclass"
        )

    def _is_unitary(self, A):
        """
        Checks whether operator A is unitary
        A can be either Qobj or ndarray
        """
        if isinstance(A, Qobj):
            unitary = np.allclose(
                np.eye(A.shape[0]),
                (A * A.dag()).full(),
                atol=self.unitarity_tol,
            )
        else:
            unitary = np.allclose(
                np.eye(len(A)), A.dot(A.T.conj()), atol=self.unitarity_tol
            )

        return unitary

    def _calc_unitary_err(self, A):
        if isinstance(A, Qobj):
            err = np.sum(abs(np.eye(A.shape[0]) - (A * A.dag()).full()))
        else:
            err = np.sum(abs(np.eye(len(A)) - A.dot(A.T.conj())))

        return err

    def unitarity_check(self):
        """
        Checks whether all propagators are unitary
        """
        for k in range(self.num_tslots):
            if not self._is_unitary(self._prop[k]):
                logger.warning(
                    "Progator of timeslot {} is not unitary".format(k)
                )

class DynamicsSymplectic(Dynamics):
    """
    Symplectic systems
    This is the subclass to use for systems where the dynamics is described
    by symplectic matrices, e.g. coupled oscillators, quantum optics

    Attributes
    ----------
    omega : array[drift_dyn_gen.shape]
        matrix used in the calculation of propagators (time evolution)
        with symplectic systems.

    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = "SYMPL"
        self._omega = None
        self._omega_qobj = None
        self._phase_application = "postop"
        self.grad_exact = True
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == "DYNAMIC":
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    @property
    def omega(self):
        if self._omega is None:
            self._get_omega()
        if self._omega_qobj is None:
            self._omega_qobj = Qobj(self._omega, dims=self.dyn_dims)
        return self._omega_qobj

    def _get_omega(self):
        if self._omega is None:
            n = self.get_drift_dim() // 2
            omg = sympl.calc_omega(n)
            if self.oper_dtype == Qobj:
                self._omega = Qobj(omg, dims=self.dyn_dims)
                self._omega_qobj = self._omega
            else:
                self._omega = omg
        return self._omega

    def _set_phase_application(self, value):
        Dynamics._set_phase_application(self, value)
        if self._evo_initialized:
            phase = self._get_dyn_gen_phase()
            if phase is not None:
                self._dyn_gen_phase = phase

    def _get_dyn_gen_phase(self):
        if self._phase_application == "postop":
            phase = -self._get_omega()
        elif self._phase_application == "preop":
            phase = self._get_omega()
        elif self._phase_application == "custom":
            phase = None
            # Assume phase set by user
        else:
            raise ValueError(
                "No option for phase_application "
                "'{}'".format(self._phase_application)
            )
        return phase

    @property
    def dyn_gen_phase(self):
        r"""
        The phasing operator for the symplectic group generators
        usually refered to as \Omega
        By default this is applied as 'postop' dyn_gen*-\Omega
        If phase_application is 'preop' it is applied as \Omega*dyn_gen
        """
        # Cannot be calculated until the dyn_shape is set
        # that is after the drift dyn gen has been set.
        if self._dyn_gen_phase is None:
            self._dyn_gen_phase = self._get_dyn_gen_phase()
        return self._dyn_gen_phase

class DynamicsGenMat(Dynamics):
    """
    This sub class can be used for any system where no additional
    operator is applied to the dynamics generator before calculating
    the propagator, e.g. classical dynamics, Lindbladian
    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = "GEN_MAT"
        self.apply_params()

class DynamicsUnitary(Dynamics):
    """
    This is the subclass to use for systems with dynamics described by
    unitary matrices. E.g. closed systems with Hermitian Hamiltonians
    Note a matrix diagonalisation is used to compute the exponent
    The eigen decomposition is also used to calculate the propagator gradient.
    The method is taken from DYNAMO (see file header)

    Attributes
    ----------
    drift_ham : Qobj
        This is the drift Hamiltonian for unitary dynamics
        It is mapped to drift_dyn_gen during initialize_controls

    ctrl_ham : List of Qobj
        These are the control Hamiltonians for unitary dynamics
        It is mapped to ctrl_dyn_gen during initialize_controls

    H : List of Qobj
        The combined drift and control Hamiltonians for each timeslot
        These are the dynamics generators for unitary dynamics.
        It is mapped to dyn_gen during initialize_controls
    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = "UNIT"
        self.drift_ham = None
        self.ctrl_ham = None
        self.H = None
        self._dyn_gen_phase = -1j
        self._phase_application = "preop"
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == "DYNAMIC":
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)
        # set the default fidelity computer
        self.fid_computer = fidcomp.FidCompUnitary(self)
        # set the default propagator computer
        self.prop_computer = propcomp.PropCompDiag(self)

    def initialize_controls(self, amplitudes, init_tslots=True):
        # Either the _dyn_gen or _ham names can be used
        # This assumes that one or other has been set in the configuration
        self._map_dyn_gen_to_ham()
        Dynamics.initialize_controls(self, amplitudes, init_tslots=init_tslots)

    def _map_dyn_gen_to_ham(self):
        if self.drift_dyn_gen is None:
            self.drift_dyn_gen = self.drift_ham
        else:
            self.drift_ham = self.drift_dyn_gen
        if self.ctrl_dyn_gen is None:
            self.ctrl_dyn_gen = self.ctrl_ham
        else:
            self.ctrl_ham = self.ctrl_dyn_gen
        self._dyn_gen_mapped = True

    @property
    def num_ctrls(self):
        if not self._dyn_gen_mapped:
            self._map_dyn_gen_to_ham()
        if self._num_ctrls is None:
            self._num_ctrls = self._get_num_ctrls()
        return self._num_ctrls

    def _get_onto_evo_target(self):
        """
        Get the adjoint of the target.
        Used for calculating the 'backward' evolution
        """
        if self.oper_dtype == Qobj:
            self._onto_evo_target = self.target.dag()
        else:
            self._onto_evo_target = self._target.T.conj()
        return self._onto_evo_target

    def _spectral_decomp(self, k):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient
        """

        if self.oper_dtype == Qobj:
            H = self._dyn_gen[k]
            # Returns eigenvalues as array (row)
            # and eigenvectors as rows of an array
            _type = _data.CSR if self.sparse_eigen_decomp else _data.Dense
            eig_val, eig_vec = _data.eigs(_data.to(_type, H.data))
            eig_vec = eig_vec.to_array()

        elif self.oper_dtype == np.ndarray:
            H = self._dyn_gen[k]
            # returns row vector of eigenvals, columns with the eigenvecs
            eig_val, eig_vec = eigh(H)

        else:
            assert False, f"Unknown oper_dtype {self.oper_dtype!r}"

        # assuming H is an nxn matrix, find n
        n = self.get_drift_dim()

        # Calculate the propagator in the diagonalised basis
        eig_val_tau = -1j * eig_val * self.tau[k]
        prop_eig = np.exp(eig_val_tau)

        # Generate the factor matrix through the differences
        # between each of the eigenvectors and the exponentiations
        # create nxn matrix where each eigen val is repeated n times
        # down the columns
        o = np.ones([n, n])
        eig_val_cols = eig_val_tau * o
        # calculate all the differences by subtracting it from its transpose
        eig_val_diffs = eig_val_cols - eig_val_cols.T
        # repeat for the propagator
        prop_eig_cols = prop_eig * o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T
        # the factor matrix is the elementwise quotient of the
        # differeneces between the exponentiated eigen vals and the
        # differences between the eigen vals
        # need to avoid division by zero that would arise due to denegerate
        # eigenvalues and the diagonals
        degen_mask = np.abs(eig_val_diffs) < self.fact_mat_round_prec
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        # for degenerate eigenvalues the factor is just the exponent
        factors[degen_mask] = prop_eig_cols[degen_mask]

        # Store eigenvectors, propagator and factor matric
        # for use in propagator computations
        self._decomp_curr[k] = True
        if isinstance(factors, np.ndarray):
            self._dyn_gen_factormatrix[k] = factors
        else:
            self._dyn_gen_factormatrix[k] = np.array(factors)

        if self.oper_dtype == Qobj:
            self._prop_eigen[k] = Qobj(
                np.diagflat(prop_eig), dims=self.dyn_dims
            )
            self._dyn_gen_eigenvectors[k] = Qobj(eig_vec, dims=self.dyn_dims)
            # The _dyn_gen_eigenvectors_adj list is not used in
            # memory optimised modes
            if self._dyn_gen_eigenvectors_adj is not None:
                self._dyn_gen_eigenvectors_adj[k] = self._dyn_gen_eigenvectors[
                    k
                ].dag()
        elif self.oper_dtype == np.ndarray:
            self._prop_eigen[k] = np.diagflat(prop_eig)
            self._dyn_gen_eigenvectors[k] = eig_vec
            # The _dyn_gen_eigenvectors_adj list is not used in
            # memory optimised modes
            if self._dyn_gen_eigenvectors_adj is not None:
                self._dyn_gen_eigenvectors_adj[k] = (
                    self._dyn_gen_eigenvectors[k].conj().T
                )
        else:
            assert False, f"Unknown oper_dtype {self.oper_dtype!r}"

    def _get_dyn_gen_eigenvectors_adj(self, k):
        # The _dyn_gen_eigenvectors_adj list is not used in
        # memory optimised modes
        if self._dyn_gen_eigenvectors_adj is not None:
            return self._dyn_gen_eigenvectors_adj[k]
        if self.oper_dtype == Qobj:
            return self._dyn_gen_eigenvectors[k].dag()
        return self._dyn_gen_eigenvectors[k].conj().T

    def check_unitarity(self):
        """
        Checks whether all propagators are unitary
        For propagators found not to be unitary, the potential underlying
        causes are investigated.
        """
        for k in range(self.num_tslots):
            prop_unit = self._is_unitary(self._prop[k])
            if not prop_unit:
                logger.warning(
                    "Progator of timeslot {} is not unitary".format(k)
                )
            if not prop_unit or self.unitarity_check_level > 1:
                # Check Hamiltonian
                H = self._dyn_gen[k]
                if isinstance(H, Qobj):
                    herm = H.isherm
                else:
                    diff = np.abs(H.T.conj() - H)
                    herm = np.all(diff < settings.core["atol"])
                eigval_unit = self._is_unitary(self._prop_eigen[k])
                eigvec_unit = self._is_unitary(self._dyn_gen_eigenvectors[k])
                if self._dyn_gen_eigenvectors_adj is not None:
                    eigvecadj_unit = self._is_unitary(
                        self._dyn_gen_eigenvectors_adj[k]
                    )
                else:
                    eigvecadj_unit = None
                msg = (
                    "prop unit: {}; H herm: {}; "
                    "eigval unit: {}; eigvec unit: {}; "
                    "eigvecadj_unit: {}".format(
                        prop_unit,
                        herm,
                        eigval_unit,
                        eigvec_unit,
                        eigvecadj_unit,
                    )
                )
                logger.info(msg)


class DynamicsSymplectic(Dynamics):
    """
    Symplectic systems
    This is the subclass to use for systems where the dynamics is described
    by symplectic matrices, e.g. coupled oscillators, quantum optics

    Attributes
    ----------
    omega : array[drift_dyn_gen.shape]
        matrix used in the calculation of propagators (time evolution)
        with symplectic systems.

    """

    def reset(self):
        Dynamics.reset(self)
        self.id_text = "SYMPL"
        self._omega = None
        self._omega_qobj = None
        self._phase_application = "postop"
        self.grad_exact = True
        self.apply_params()

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to _UpdateAll
        # can be set to _DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == "DYNAMIC":
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)

    @property
    def omega(self):
        if self._omega is None:
            self._get_omega()
        if self._omega_qobj is None:
            self._omega_qobj = Qobj(self._omega, dims=self.dyn_dims)
        return self._omega_qobj

    def _get_omega(self):
        if self._omega is None:
            n = self.get_drift_dim() // 2
            omg = sympl.calc_omega(n)
            if self.oper_dtype == Qobj:
                self._omega = Qobj(omg, dims=self.dyn_dims)
                self._omega_qobj = self._omega
            else:
                self._omega = omg
        return self._omega

    def _set_phase_application(self, value):
        Dynamics._set_phase_application(self, value)
        if self._evo_initialized:
            phase = self._get_dyn_gen_phase()
            if phase is not None:
                self._dyn_gen_phase = phase

    def _get_dyn_gen_phase(self):
        if self._phase_application == "postop":
            phase = -self._get_omega()
        elif self._phase_application == "preop":
            phase = self._get_omega()
        elif self._phase_application == "custom":
            phase = None
            # Assume phase set by user
        else:
            raise ValueError(
                "No option for phase_application "
                "'{}'".format(self._phase_application)
            )
        return phase

    @property
    def dyn_gen_phase(self):
        r"""
        The phasing operator for the symplectic group generators
        usually refered to as \Omega
        By default this is applied as 'postop' dyn_gen*-\Omega
        If phase_application is 'preop' it is applied as \Omega*dyn_gen
        """
        # Cannot be calculated until the dyn_shape is set
        # that is after the drift dyn gen has been set.
        if self._dyn_gen_phase is None:
            self._dyn_gen_phase = self._get_dyn_gen_phase()
        return self._dyn_gen_phase
    
if _parse_version(scipy.__version__) < _parse_version("1.5"):

    @functools.wraps(spopt.fmin_l_bfgs_b)
    def fmin_l_bfgs_b(*args, **kwargs):
        with warnings.catch_warnings():
            message = r"tostring\(\) is deprecated\. Use tobytes\(\) instead\."
            warnings.filterwarnings(
                "ignore", message=message, category=DeprecationWarning
            )
            return spopt.fmin_l_bfgs_b(*args, **kwargs)

else:
    fmin_l_bfgs_b = spopt.fmin_l_bfgs_b


def _is_string(var):
    try:
        if isinstance(var, basestring):
            return True
    except NameError:
        try:
            if isinstance(var, str):
                return True
        except:
            return False
    except:
        return False


class Optimizer(object):
    """
    Base class for all control pulse optimisers. This class should not be
    instantiated, use its subclasses.  This class implements the fidelity,
    gradient and interation callback functions.  All subclass objects must be
    initialised with a

    - ``OptimConfig`` instance - various configuration options
    - ``Dynamics`` instance - describes the dynamics of the (quantum) system
      to be control optimised

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.  Options are attributes of
        qutip_qtrl.logging_utils, in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution, assuming
        everything runs as expected.  The default NOTSET implies that the level
        will be taken from the QuTiP settings file, which by default is WARN.

    params:  Dictionary
        The key value pairs are the attribute name and value. Note: attributes
        are created if they do not exist already, and are overwritten if they
        do.

    alg : string
        Algorithm to use in pulse optimisation.  Options are:

        - 'GRAPE' (default) - GRadient Ascent Pulse Engineering
        - 'CRAB' - Chopped RAndom Basis

    alg_params : Dictionary
        Options that are specific to the pulse optim algorithm ``alg``.

    disp_conv_msg : bool
        Set true to display a convergence message
        (for scipy.optimize.minimize methods anyway)

    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error

    method_params : Dictionary
        Options for the optim_method.
        Note that where there is an equivalent attribute of this instance
        or the termination_conditions (for example maxiter)
        it will override an value in these options

    approx_grad : bool
        If set True then the method will approximate the gradient itself
        (if it has requirement and facility for this)
        This will mean that the fid_err_grad_wrapper will not get called
        Note it should be left False when using the Dynamics
        to calculate approximate gradients
        Note it is set True automatically when the alg is CRAB

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    bounds : List of floats
        Bounds for the parameters.
        If not set before the run_optimization call then the list
        is built automatically based on the amp_lbound and amp_ubound
        attributes.
        Setting this attribute directly allows specific bounds to be set
        for individual parameters.
        Note: Only some methods use bounds

    dynamics : Dynamics (subclass instance)
        describes the dynamics of the (quantum) system to be control optimised
        (see Dynamics classes for details)

    config : OptimConfig instance
        various configuration options
        (see OptimConfig for details)

    termination_conditions : TerminationCondition instance
        attributes determine when the optimisation will end

    pulse_generator : PulseGen (subclass instance)
        (can be) used to create initial pulses
        not used by the class, but set by pulseoptim.create_pulse_optimizer

    stats : Stats
        attributes of which give performance stats for the optimisation
        set to None to reduce overhead of calculating stats.
        Note it is (usually) shared with the Dynamics instance

    dump : :class:`qutip_qtrl.dump.OptimDump`
        Container for data dumped during the optimisation.
        Can be set by specifying the dumping level or set directly.
        Note this is mainly intended for user and a development debugging
        but could be used for status information during a long optimisation.

    dumping : string
        level of data dumping: NONE, SUMMARY, FULL or CUSTOM
        See property docstring for details

    dump_to_file : bool
        If set True then data will be dumped to file during the optimisation
        dumping will be set to SUMMARY during init_optim
        if dump_to_file is True and dumping not set.
        Default is False

    dump_dir : string
        Basically a link to dump.dump_dir. Exists so that it can be set through
        optim_params.
        If dump is None then will return None or will set dumping to SUMMARY
        when setting a path

    iter_summary : :class:`OptimIterSummary`
        Summary of the most recent iteration.
        Note this is only set if dummping is on
    """

    def __init__(self, config, dyn, params=None):
        self.dynamics = dyn
        self.config = config
        self.params = params
        self.reset()
        dyn.parent = self

    def reset(self):
        self.log_level = self.config.log_level
        self.id_text = "OPTIM"
        self.termination_conditions = None
        self.pulse_generator = None
        self.disp_conv_msg = False
        self.iteration_steps = None
        self.record_iteration_steps = False
        self.alg = "GRAPE"
        self.alg_params = None
        self.method = "l_bfgs_b"
        self.method_params = None
        self.method_options = None
        self.approx_grad = False
        self.amp_lbound = None
        self.amp_ubound = None
        self.bounds = None
        self.num_iter = 0
        self.num_fid_func_calls = 0
        self.num_grad_func_calls = 0
        self.stats = None
        self.wall_time_optim_start = 0.0

        self.dump_to_file = False
        self.dump = None
        self.iter_summary = None

        # AJGP 2015-04-21:
        # These (copying from config) are here for backward compatibility
        if hasattr(self.config, "amp_lbound"):
            if self.config.amp_lbound:
                self.amp_lbound = self.config.amp_lbound
        if hasattr(self.config, "amp_ubound"):
            if self.config.amp_ubound:
                self.amp_ubound = self.config.amp_ubound

        self.apply_params()

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def dumping(self):
        """
        The level of data dumping that will occur during the optimisation

        - NONE : No processing data dumped (Default)
        - SUMMARY : A summary at each iteration will be recorded
        - FULL : All logs will be generated and dumped
        - CUSTOM : Some customised level of dumping

        When first set to CUSTOM this is equivalent to SUMMARY. It is then up
        to the user to specify which logs are dumped
        """
        if self.dump is None:
            lvl = "NONE"
        else:
            lvl = self.dump.level

        return lvl

    @dumping.setter
    def dumping(self, value):
        if value is None:
            self.dump = None
        else:
            if not isinstance(value, str):
                raise TypeError("Value must be string value")
            lvl = value.upper()
            if lvl == "NONE":
                self.dump = None
            else:
                if not isinstance(self.dump, qtrldump.OptimDump):
                    self.dump = qtrldump.OptimDump(self, level=lvl)
                else:
                    self.dump.level = lvl

    @property
    def dump_dir(self):
        if self.dump:
            return self.dump.dump_dir
        else:
            return None

    @dump_dir.setter
    def dump_dir(self, value):
        if not self.dump:
            self.dumping = "SUMMARY"
        self.dump.dump_dir = value

    def _create_result(self):
        """
        create the result object
        and set the initial_amps attribute as the current amplitudes
        """
        result = optimresult.OptimResult()
        result.initial_fid_err = self.dynamics.fid_computer.get_fid_err()
        result.initial_amps = self.dynamics.ctrl_amps.copy()
        result.evo_full_initial = self.dynamics.full_evo.copy()
        result.time = self.dynamics.time.copy()
        result.optimizer = self
        return result

    def init_optim(self, term_conds):
        """
        Check optimiser attribute status and passed parameters before
        running the optimisation.
        This is called by run_optimization, but could called independently
        to check the configuration.
        """
        if term_conds is not None:
            self.termination_conditions = term_conds
        term_conds = self.termination_conditions

        if not isinstance(term_conds, termcond.TerminationConditions):
            raise errors.UsageError(
                "No termination conditions for the " "optimisation function"
            )

        if not isinstance(self.dynamics, dynamics.Dynamics):
            raise errors.UsageError("No dynamics object attribute set")
        self.dynamics.check_ctrls_initialized()

        self.apply_method_params()

        if term_conds.fid_err_targ is None and term_conds.fid_goal is None:
            raise errors.UsageError(
                "Either the goal or the fidelity "
                "error tolerance must be set"
            )

        if term_conds.fid_err_targ is None:
            term_conds.fid_err_targ = np.abs(1 - term_conds.fid_goal)

        if term_conds.fid_goal is None:
            term_conds.fid_goal = 1 - term_conds.fid_err_targ

        if self.alg == "CRAB":
            self.approx_grad = True

        if self.stats is not None:
            self.stats.clear()

        if self.dump_to_file:
            if self.dump is None:
                self.dumping = "SUMMARY"
            self.dump.write_to_file = True
            self.dump.create_dump_dir()
            logger.info(
                "Optimiser dump will be written to:\n{}".format(
                    self.dump.dump_dir
                )
            )

        if self.dump:
            self.iter_summary = OptimIterSummary()
        else:
            self.iter_summary = None

        self.num_iter = 0
        self.num_fid_func_calls = 0
        self.num_grad_func_calls = 0
        self.iteration_steps = None

    def _build_method_options(self):
        """
        Creates the method_options dictionary for the scipy.optimize.minimize
        function based on the attributes of this object and the
        termination_conditions
        It assumes that apply_method_params has already been run and
        hence the method_options attribute may already contain items.
        These values will NOT be overridden
        """
        tc = self.termination_conditions
        if self.method_options is None:
            self.method_options = {}
        mo = self.method_options

        if "max_metric_corr" in mo and "maxcor" not in mo:
            mo["maxcor"] = mo["max_metric_corr"]
        elif hasattr(self, "max_metric_corr") and "maxcor" not in mo:
            mo["maxcor"] = self.max_metric_corr
        if "accuracy_factor" in mo and "ftol" not in mo:
            mo["ftol"] = mo["accuracy_factor"]
        elif hasattr(tc, "accuracy_factor") and "ftol" not in mo:
            mo["ftol"] = tc.accuracy_factor
        if tc.max_iterations > 0 and "maxiter" not in mo:
            mo["maxiter"] = tc.max_iterations
        if tc.max_fid_func_calls > 0 and "maxfev" not in mo:
            mo["maxfev"] = tc.max_fid_func_calls
        if tc.min_gradient_norm > 0 and "gtol" not in mo:
            mo["gtol"] = tc.min_gradient_norm
        if "disp" not in mo:
            mo["disp"] = self.disp_conv_msg
        return mo

    def apply_method_params(self, params=None):
        """
        Loops through all the method_params
        (either passed here or the method_params attribute)
        If the name matches an attribute of this object or the
        termination conditions object, then the value of this attribute
        is set. Otherwise it is assumed to a method_option for the
        scipy.optimize.minimize function
        """
        if not params:
            params = self.method_params

        if isinstance(params, dict):
            self.method_params = params
            unused_params = {}
            for key in params:
                val = params[key]
                if hasattr(self, key):
                    setattr(self, key, val)
                if hasattr(self.termination_conditions, key):
                    setattr(self.termination_conditions, key, val)
                else:
                    unused_params[key] = val

            if len(unused_params) > 0:
                if not isinstance(self.method_options, dict):
                    self.method_options = unused_params
                else:
                    self.method_options.update(unused_params)

    def _build_bounds_list(self):
        cfg = self.config
        dyn = self.dynamics
        n_ctrls = dyn.num_ctrls
        self.bounds = []
        for t in range(dyn.num_tslots):
            for c in range(n_ctrls):
                if isinstance(self.amp_lbound, list):
                    lb = self.amp_lbound[c]
                else:
                    lb = self.amp_lbound
                if isinstance(self.amp_ubound, list):
                    ub = self.amp_ubound[c]
                else:
                    ub = self.amp_ubound

                if lb is not None and np.isinf(lb):
                    lb = None
                if ub is not None and np.isinf(ub):
                    ub = None

                self.bounds.append((lb, ub))

    def run_optimization(self, term_conds=None):
        """
        This default function optimisation method is a wrapper to the
        scipy.optimize.minimize function.

        It will attempt to minimise the fidelity error with respect to some
        parameters, which are determined by _get_optim_var_vals (see below)

        The optimisation end when one of the passed termination conditions
        has been met, e.g. target achieved, wall time, or
        function call or iteration count exceeded. Note these
        conditions include gradient minimum met (local minima) for
        methods that use a gradient.

        The function minimisation method is taken from the optim_method
        attribute. Note that not all of these methods have been tested.
        Note that some of these use a gradient and some do not.
        See the scipy documentation for details. Options specific to the
        method can be passed setting the method_params attribute.

        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None

        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc
        """
        self.init_optim(term_conds)
        term_conds = self.termination_conditions
        dyn = self.dynamics
        cfg = self.config
        self.optim_var_vals = self._get_optim_var_vals()
        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time

        if self.stats is not None:
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 0

        if self.bounds is None:
            self._build_bounds_list()

        self._build_method_options()

        result = self._create_result()

        if self.approx_grad:
            jac = None
        else:
            jac = self.fid_err_grad_wrapper

        if self.log_level <= logging.INFO:
            msg = (
                "Optimising pulse(s) using {} with " "minimise '{}' method"
            ).format(self.alg, self.method)
            if self.approx_grad:
                msg += " (approx grad)"
            logger.info(msg)

        try:
            opt_res = spopt.minimize(
                self.fid_err_func_wrapper,
                self.optim_var_vals,
                method=self.method,
                jac=jac,
                bounds=self.bounds,
                options=self.method_options,
                callback=self.iter_step_callback_func,
            )

            amps = self._get_ctrl_amps(opt_res.x)
            dyn.update_ctrl_amps(amps)
            result.termination_reason = opt_res.message
            # Note the iterations are counted in this object as well
            # so there are compared here for interest sake only
            if self.num_iter != opt_res.nit:
                logger.info(
                    "The number of iterations counted {} "
                    " does not match the number reported {} "
                    "by {}".format(self.num_iter, opt_res.nit, self.method)
                )
            result.num_iter = opt_res.nit

        except errors.OptimizationTerminate as except_term:
            self._interpret_term_exception(except_term, result)

        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)

        return result

    def _get_optim_var_vals(self):
        """
        Generate the 1d array that holds the current variable values
        of the function to be optimised
        By default (as used in GRAPE) these are the control amplitudes
        in each timeslot
        """
        return self.dynamics.ctrl_amps.reshape([-1])

    def _get_ctrl_amps(self, optim_var_vals):
        """
        Get the control amplitudes from the current variable values
        of the function to be optimised.
        that is the 1d array that is passed from the optimisation method
        Note for GRAPE these are the function optimiser parameters
        (and this is the default)

        Returns
        -------
        float array[dynamics.num_tslots, dynamics.num_ctrls]
        """
        amps = optim_var_vals.reshape(self.dynamics.ctrl_amps.shape)

        return amps

    def fid_err_func_wrapper(self, *args):
        """
        Get the fidelity error achieved using the ctrl amplitudes passed
        in as the first argument.

        This is called by generic optimisation algorithm as the
        func to the minimised. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)

        The error is checked against the target, and the optimisation is
        terminated if the target has been achieved.
        """
        self.num_fid_func_calls += 1
        # *** update stats ***
        if self.stats is not None:
            self.stats.num_fidelity_func_calls = self.num_fid_func_calls
            if self.log_level <= logging.DEBUG:
                logger.debug(
                    "fidelity error call {}".format(
                        self.stats.num_fidelity_func_calls
                    )
                )

        amps = self._get_ctrl_amps(args[0].copy())
        self.dynamics.update_ctrl_amps(amps)

        tc = self.termination_conditions
        err = self.dynamics.fid_computer.get_fid_err()

        if self.iter_summary:
            self.iter_summary.fid_func_call_num = self.num_fid_func_calls
            self.iter_summary.fid_err = err

        if self.dump and self.dump.dump_fid_err:
            self.dump.update_fid_err_log(err)

        if err <= tc.fid_err_targ:
            raise errors.GoalAchievedTerminate(err)

        if self.num_fid_func_calls > tc.max_fid_func_calls:
            raise errors.MaxFidFuncCallTerminate()

        return err

    def fid_err_grad_wrapper(self, *args):
        """
        Get the gradient of the fidelity error with respect to all of the
        variables, i.e. the ctrl amplidutes in each timeslot

        This is called by generic optimisation algorithm as the gradients of
        func to the minimised wrt the variables. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)

        Although the optimisation algorithms have a check within them for
        function convergence, i.e. local minima, the sum of the squares
        of the normalised gradient is checked explicitly, and the
        optimisation is terminated if this is below the min_gradient_norm
        condition
        """
        # *** update stats ***
        self.num_grad_func_calls += 1
        if self.stats is not None:
            self.stats.num_grad_func_calls = self.num_grad_func_calls
            if self.log_level <= logging.DEBUG:
                logger.debug(
                    "gradient call {}".format(self.stats.num_grad_func_calls)
                )
        amps = self._get_ctrl_amps(args[0].copy())
        self.dynamics.update_ctrl_amps(amps)
        fid_comp = self.dynamics.fid_computer
        # gradient_norm_func is a pointer to the function set in the config
        # that returns the normalised gradients
        grad = fid_comp.get_fid_err_gradient()

        if self.iter_summary:
            self.iter_summary.grad_func_call_num = self.num_grad_func_calls
            self.iter_summary.grad_norm = fid_comp.grad_norm

        if self.dump:
            if self.dump.dump_grad_norm:
                self.dump.update_grad_norm_log(fid_comp.grad_norm)

            if self.dump.dump_grad:
                self.dump.update_grad_log(grad)

        tc = self.termination_conditions
        if fid_comp.grad_norm < tc.min_gradient_norm:
            raise errors.GradMinReachedTerminate(fid_comp.grad_norm)
        return grad.flatten()

    def iter_step_callback_func(self, *args):
        """
        Check the elapsed wall time for the optimisation run so far.
        Terminate if this has exceeded the maximum allowed time
        """
        self.num_iter += 1

        if self.log_level <= logging.DEBUG:
            logger.debug("Iteration callback {}".format(self.num_iter))

        wall_time = timeit.default_timer() - self.wall_time_optimize_start

        if self.iter_summary:
            self.iter_summary.iter_num = self.num_iter
            self.iter_summary.wall_time = wall_time

        if self.dump and self.dump.dump_summary:
            self.dump.add_iter_summary()

        tc = self.termination_conditions

        if wall_time > tc.max_wall_time:
            raise errors.MaxWallTimeTerminate()

        # *** update stats ***
        if self.stats is not None:
            self.stats.num_iter = self.num_iter

    def _interpret_term_exception(self, except_term, result):
        """
        Update the result object based on the exception that occurred
        during the optimisation
        """
        result.termination_reason = except_term.reason
        if isinstance(except_term, errors.GoalAchievedTerminate):
            result.goal_achieved = True
        elif isinstance(except_term, errors.MaxWallTimeTerminate):
            result.wall_time_limit_exceeded = True
        elif isinstance(except_term, errors.GradMinReachedTerminate):
            result.grad_norm_min_reached = True
        elif isinstance(except_term, errors.MaxFidFuncCallTerminate):
            result.max_fid_func_exceeded = True

    def _add_common_result_attribs(self, result, st_time, end_time):
        """
        Update the result object attributes which are common to all
        optimisers and outcomes
        """
        dyn = self.dynamics
        result.num_iter = self.num_iter
        result.num_fid_func_calls = self.num_fid_func_calls
        result.wall_time = end_time - st_time
        result.fid_err = dyn.fid_computer.get_fid_err()
        result.grad_norm_final = dyn.fid_computer.grad_norm
        result.final_amps = dyn.ctrl_amps
        final_evo = dyn.full_evo
        if isinstance(final_evo, Qobj):
            result.evo_full_final = final_evo
        else:
            result.evo_full_final = Qobj(final_evo, dims=dyn.sys_dims)
        # *** update stats ***
        if self.stats is not None:
            self.stats.wall_time_optim_end = end_time
            self.stats.calculate()
            result.stats = copy.copy(self.stats)


class OptimizerBFGS(Optimizer):
    """
    Implements the run_optimization method using the BFGS algorithm
    """

    def reset(self):
        Optimizer.reset(self)
        self.id_text = "BFGS"

    def run_optimization(self, term_conds=None):
        """
        Optimise the control pulse amplitudes to minimise the fidelity error
        using the BFGS (Broyden–Fletcher–Goldfarb–Shanno) algorithm
        The optimisation end when one of the passed termination conditions
        has been met, e.g. target achieved, gradient minimum met
        (local minima), wall time / iteration count exceeded.

        Essentially this is wrapper to the:
        scipy.optimize.fmin_bfgs
        function

        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None

        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc
        """
        self.init_optim(term_conds)
        term_conds = self.termination_conditions
        dyn = self.dynamics
        self.optim_var_vals = self._get_optim_var_vals()
        self._build_method_options()

        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time

        if self.stats is not None:
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 1

        if self.approx_grad:
            fprime = None
        else:
            fprime = self.fid_err_grad_wrapper

        if self.log_level <= logging.INFO:
            msg = (
                "Optimising pulse(s) using {} with " "'fmin_bfgs' method"
            ).format(self.alg)
            if self.approx_grad:
                msg += " (approx grad)"
            logger.info(msg)

        result = self._create_result()
        try:
            (
                optim_var_vals,
                cost,
                grad,
                invHess,
                nFCalls,
                nGCalls,
                warn,
            ) = spopt.fmin_bfgs(
                self.fid_err_func_wrapper,
                self.optim_var_vals,
                fprime=fprime,
                callback=self.iter_step_callback_func,
                gtol=term_conds.min_gradient_norm,
                maxiter=term_conds.max_iterations,
                full_output=True,
                disp=True,
            )

            amps = self._get_ctrl_amps(optim_var_vals)
            dyn.update_ctrl_amps(amps)
            if warn == 1:
                result.max_iter_exceeded = True
                result.termination_reason = "Iteration count limit reached"
            elif warn == 2:
                result.grad_norm_min_reached = True
                result.termination_reason = "Gradient normal minimum reached"

        except errors.OptimizationTerminate as except_term:
            self._interpret_term_exception(except_term, result)

        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)

        return result


class OptimizerLBFGSB(Optimizer):
    """
    Implements the run_optimization method using the L-BFGS-B algorithm

    Attributes
    ----------
    max_metric_corr : integer
        The maximum number of variable metric corrections used to define
        the limited memory matrix. That is the number of previous
        gradient values that are used to approximate the Hessian
        see the scipy.optimize.fmin_l_bfgs_b documentation for description
        of m argument
    """

    def reset(self):
        Optimizer.reset(self)
        self.id_text = "LBFGSB"
        self.max_metric_corr = 10
        self.msg_level = None

    def init_optim(self, term_conds):
        """
        Check optimiser attribute status and passed parameters before
        running the optimisation.
        This is called by run_optimization, but could called independently
        to check the configuration.
        """
        if term_conds is None:
            term_conds = self.termination_conditions

        # AJGP 2015-04-21:
        # These (copying from config) are here for backward compatibility
        if hasattr(self.config, "max_metric_corr"):
            if self.config.max_metric_corr:
                self.max_metric_corr = self.config.max_metric_corr
        if hasattr(self.config, "accuracy_factor"):
            if self.config.accuracy_factor:
                term_conds.accuracy_factor = self.config.accuracy_factor

        Optimizer.init_optim(self, term_conds)

        if not isinstance(self.msg_level, int):
            if self.log_level < logging.DEBUG:
                self.msg_level = 2
            elif self.log_level <= logging.DEBUG:
                self.msg_level = 1
            else:
                self.msg_level = 0

    def run_optimization(self, term_conds=None):
        """
        Optimise the control pulse amplitudes to minimise the fidelity error
        using the L-BFGS-B algorithm, which is the constrained
        (bounded amplitude values), limited memory, version of the
        Broyden–Fletcher–Goldfarb–Shanno algorithm.

        The optimisation end when one of the passed termination conditions
        has been met, e.g. target achieved, gradient minimum met
        (local minima), wall time / iteration count exceeded.

        Essentially this is wrapper to the:
        scipy.optimize.fmin_l_bfgs_b function
        This in turn is a warpper for well established implementation of
        the L-BFGS-B algorithm written in Fortran, which is therefore
        very fast. See SciPy documentation for credit and details on
        this function.

        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None

        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc
        """
        self.init_optim(term_conds)
        term_conds = self.termination_conditions
        dyn = self.dynamics
        cfg = self.config
        self.optim_var_vals = self._get_optim_var_vals()
        self._build_method_options()

        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time

        if self.stats is not None:
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 1

        bounds = self._build_bounds_list()
        result = self._create_result()

        if self.approx_grad:
            fprime = None
        else:
            fprime = self.fid_err_grad_wrapper

        if "accuracy_factor" in self.method_options:
            factr = self.method_options["accuracy_factor"]
        elif "ftol" in self.method_options:
            factr = self.method_options["ftol"]
        elif hasattr(term_conds, "accuracy_factor"):
            factr = term_conds.accuracy_factor
        else:
            factr = 1e7

        if "max_metric_corr" in self.method_options:
            m = self.method_options["max_metric_corr"]
        elif "maxcor" in self.method_options:
            m = self.method_options["maxcor"]
        elif hasattr(self, "max_metric_corr"):
            m = self.max_metric_corr
        else:
            m = 10

        if self.log_level <= logging.INFO:
            msg = (
                "Optimising pulse(s) using {} with " "'fmin_l_bfgs_b' method"
            ).format(self.alg)
            if self.approx_grad:
                msg += " (approx grad)"
            logger.info(msg)
        try:
            optim_var_vals, fid, res_dict = fmin_l_bfgs_b(
                self.fid_err_func_wrapper,
                self.optim_var_vals,
                fprime=fprime,
                approx_grad=self.approx_grad,
                callback=self.iter_step_callback_func,
                bounds=self.bounds,
                m=m,
                factr=factr,
                pgtol=term_conds.min_gradient_norm,
                disp=self.msg_level,
                maxfun=term_conds.max_fid_func_calls,
                maxiter=term_conds.max_iterations,
            )

            amps = self._get_ctrl_amps(optim_var_vals)
            dyn.update_ctrl_amps(amps)
            warn = res_dict["warnflag"]
            if warn == 0:
                result.grad_norm_min_reached = True
                result.termination_reason = "function converged"
            elif warn == 1:
                result.max_iter_exceeded = True
                result.termination_reason = (
                    "Iteration or fidelity " "function call limit reached"
                )
            elif warn == 2:
                result.termination_reason = res_dict["task"]

            result.num_iter = res_dict["nit"]
        except errors.OptimizationTerminate as except_term:
            self._interpret_term_exception(except_term, result)

        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)

        return result


class OptimizerCrab(Optimizer):
    """
    Optimises the pulse using the CRAB algorithm [Caneva]_.
    It uses the scipy.optimize.minimize function with the method specified
    by the optim_method attribute. See Optimizer.run_optimization for details
    It minimises the fidelity error function with respect to the CRAB
    basis function coefficients.

    References
    ----------
    .. [Caneva] T. Caneva, T. Calarco, and S. Montangero. Chopped random-basis quantum optimization,
       Phys. Rev. A, 84:022326, 2011 (doi:10.1103/PhysRevA.84.022326)
    """

    def reset(self):
        Optimizer.reset(self)
        self.id_text = "CRAB"
        self.num_optim_vars = 0

    def init_optim(self, term_conds):
        """
        Check optimiser attribute status and passed parameters before
        running the optimisation.
        This is called by run_optimization, but could called independently
        to check the configuration.
        """
        Optimizer.init_optim(self, term_conds)
        dyn = self.dynamics

        self.num_optim_vars = 0
        pulse_gen_valid = True
        # check the pulse generators match the ctrls
        # (in terms of number)
        # and count the number of parameters
        if self.pulse_generator is None:
            pulse_gen_valid = False
            err_msg = "pulse_generator attribute is None"
        elif not isinstance(self.pulse_generator, collections.abc.Iterable):
            pulse_gen_valid = False
            err_msg = "pulse_generator is not iterable"

        elif len(self.pulse_generator) != dyn.num_ctrls:
            pulse_gen_valid = False
            err_msg = (
                "the number of pulse generators {} does not equal "
                "the number of controls {}".format(
                    len(self.pulse_generator), dyn.num_ctrls
                )
            )

        if pulse_gen_valid:
            for p_gen in self.pulse_generator:
                if not isinstance(p_gen, pulsegen.PulseGenCrab):
                    pulse_gen_valid = False
                    err_msg = (
                        "pulse_generator contained object of type '{}'".format(
                            p_gen.__class__.__name__
                        )
                    )
                    break
                self.num_optim_vars += p_gen.num_optim_vars

        if not pulse_gen_valid:
            raise errors.UsageError(
                "The pulse_generator attribute must be set to a list of "
                "PulseGenCrab - one for each control. Here " + err_msg
            )

    def _build_bounds_list(self):
        """
        No bounds necessary here, as the bounds for the CRAB parameters
        do not have much physical meaning.
        This needs to override the default method, otherwise the shape
        will be wrong
        """
        return None

    def _get_optim_var_vals(self):
        """
        Generate the 1d array that holds the current variable values
        of the function to be optimised
        For CRAB these are the basis coefficients

        Returns
        -------
        ndarray (1d) of float
        """
        pvals = []
        for pgen in self.pulse_generator:
            pvals.extend(pgen.get_optim_var_vals())

        return np.array(pvals)

    def _get_ctrl_amps(self, optim_var_vals):
        """
        Get the control amplitudes from the current variable values
        of the function to be optimised.
        that is the 1d array that is passed from the optimisation method
        For CRAB the amplitudes will need to calculated by expanding the
        series

        Returns
        -------
        float array[dynamics.num_tslots, dynamics.num_ctrls]
        """
        dyn = self.dynamics

        if self.log_level <= logging.DEBUG:
            changed_params = self.optim_var_vals != optim_var_vals
            logger.debug(
                "{} out of {} optimisation parameters changed".format(
                    changed_params.sum(), len(optim_var_vals)
                )
            )

        amps = np.empty([dyn.num_tslots, dyn.num_ctrls])
        j = 0
        param_idx_st = 0
        for p_gen in self.pulse_generator:
            param_idx_end = param_idx_st + p_gen.num_optim_vars
            pg_pvals = optim_var_vals[param_idx_st:param_idx_end]
            p_gen.set_optim_var_vals(pg_pvals)
            amps[:, j] = p_gen.gen_pulse()
            param_idx_st = param_idx_end
            j += 1

        self.optim_var_vals = optim_var_vals
        return amps


class OptimizerCrabFmin(OptimizerCrab):
    """
    Optimises the pulse using the CRAB algorithm [Doria]_, [Caneva]_.
    It uses the ``scipy.optimize.fmin`` function which is effectively a wrapper
    for the Nelder-Mead method.  It minimises the fidelity error function with
    respect to the CRAB basis function coefficients.  This is the default
    Optimizer for CRAB.

    References
    ----------
    .. [Doria] P. Doria, T. Calarco & S. Montangero. Phys. Rev. Lett. 106, 190501
       (2011).
    .. [Caneva] T. Caneva, T. Calarco, & S. Montangero. Phys. Rev. A 84, 022326
       (2011).
    """

    def reset(self):
        OptimizerCrab.reset(self)
        self.id_text = "CRAB_FMIN"
        self.xtol = 1e-4
        self.ftol = 1e-4

    def run_optimization(self, term_conds=None):
        """
        This function optimisation method is a wrapper to the
        scipy.optimize.fmin function.

        It will attempt to minimise the fidelity error with respect to some
        parameters, which are determined by _get_optim_var_vals which
        in the case of CRAB are the basis function coefficients

        The optimisation end when one of the passed termination conditions
        has been met, e.g. target achieved, wall time, or
        function call or iteration count exceeded. Specifically to the fmin
        method, the optimisation will stop when change parameter values
        is less than xtol or the change in function value is below ftol.

        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None

        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc
        """
        self.init_optim(term_conds)
        term_conds = self.termination_conditions
        dyn = self.dynamics
        cfg = self.config
        self.optim_var_vals = self._get_optim_var_vals()
        self._build_method_options()

        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time

        if self.stats is not None:
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 1

        result = self._create_result()

        if self.log_level <= logging.INFO:
            logger.info(
                "Optimising pulse(s) using {} with "
                "'fmin' (Nelder-Mead) method".format(self.alg)
            )

        try:
            ret = spopt.fmin(
                self.fid_err_func_wrapper,
                self.optim_var_vals,
                xtol=self.xtol,
                ftol=self.ftol,
                maxiter=term_conds.max_iterations,
                maxfun=term_conds.max_fid_func_calls,
                full_output=True,
                disp=self.disp_conv_msg,
                retall=self.record_iteration_steps,
                callback=self.iter_step_callback_func,
            )

            final_param_vals = ret[0]
            num_iter = ret[2]
            warn_flag = ret[4]
            if self.record_iteration_steps:
                self.iteration_steps = ret[5]
            amps = self._get_ctrl_amps(final_param_vals)
            dyn.update_ctrl_amps(amps)

            # Note the iterations are counted in this object as well
            # so there are compared here for interest sake only
            if self.num_iter != num_iter:
                logger.info(
                    "The number of iterations counted {} "
                    " does not match the number reported {} "
                    "by {}".format(self.num_iter, num_iter, self.method)
                )
            result.num_iter = num_iter
            if warn_flag == 0:
                result.termination_reason = (
                    "Function converged (within tolerance)"
                )
            elif warn_flag == 1:
                result.termination_reason = (
                    "Maximum number of function evaluations reached"
                )
                result.max_fid_func_exceeded = True
            elif warn_flag == 2:
                result.termination_reason = (
                    "Maximum number of iterations reached"
                )
                result.max_iter_exceeded = True
            else:
                result.termination_reason = "Unknown (warn_flag={})".format(
                    warn_flag
                )

        except errors.OptimizationTerminate as except_term:
            self._interpret_term_exception(except_term, result)

        end_time = timeit.default_timer()
        self._add_common_result_attribs(result, st_time, end_time)

        return result

def create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        alg='GRAPE', alg_params=None,
        optim_params=None, optim_method='DEF', method_params=None,
        optim_alg=None, max_metric_corr=None, accuracy_factor=None,
        dyn_type='GEN_MAT', dyn_params=None,
        prop_type='DEF', prop_params=None,
        fid_type='DEF', fid_params=None,
        phase_option=None, fid_err_scale_factor=None,
        tslot_type='DEF', tslot_params=None,
        amp_update_mode=None,
        init_pulse_type='DEF', init_pulse_params=None,
        pulse_scaling=1.0, pulse_offset=0.0,
        ramping_pulse_type=None, ramping_pulse_params=None,
        log_level=logging.NOTSET, gen_stats=False):

    """
    Generate the objects of the appropriate subclasses
    required for the pulse optmisation based on the parameters given
    Note this method may be preferable to calling optimize_pulse
    if more detailed configuration is required before running the
    optmisation algorthim, or the algorithm will be run many times,
    for instances when trying to finding global the optimum or
    minimum time optimisation

    Parameters
    ----------
    drift : Qobj or list of Qobj
        the underlying dynamics generator of the system
        can provide list (of length num_tslots) for time dependent drift

    ctrls : List of Qobj
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the optimisation algorithm
        
    alg : string
        Algorithm to use in pulse optimisation.
        Options are:
            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    alg_params : Dictionary
        options that are specific to the algorithm see above
        
    optim_params : Dictionary
        The key value pairs are the attribute name and value
        used to set attribute values
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        Note: method_params are applied afterwards and so may override these
        
    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error
        Note that FMIN, FMIN_BFGS & FMIN_L_BFGS_B will all result
        in calling these specific scipy.optimize methods
        Note the LBFGSB is equivalent to FMIN_L_BFGS_B for backwards 
        capatibility reasons.
        Supplying DEF will given alg dependent result:
            - GRAPE - Default optim_method is FMIN_L_BFGS_B
            - CRAB - Default optim_method is Nelder-Mead
        
    method_params : dict
        Parameters for the optim_method. 
        Note that where there is an attribute of the
        Optimizer object or the termination_conditions matching the key 
        that attribute. Otherwise, and in some case also, 
        they are assumed to be method_options
        for the scipy.optimize.minimize method.        
        
    optim_alg : string
        Deprecated. Use optim_method.

    max_metric_corr : integer
        Deprecated. Use method_params instead

    accuracy_factor : float
        Deprecated. Use method_params instead

    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)
        
    dyn_params : dict
        Parameters for the Dynamics object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    prop_params : dict
        Parameters for the PropagatorComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created
        
    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    fid_params : dict
        Parameters for the FidelityComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created
        
    phase_option : string
        Deprecated. Pass in fid_params instead.

    fid_err_scale_factor : float
        Deprecated. Use scale_factor key in fid_params instead.

    tslot_type : string
        Method for computing the dynamics generators, propagators and 
        evolution in the timeslots.
        Options: DEF, UPDATE_ALL, DYNAMIC
        UPDATE_ALL is the only one that currently works
        (See TimeslotComputer classes for details)
        
    tslot_params : dict
        Parameters for the TimeslotComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created
        
    amp_update_mode : string
        Deprecated. Use tslot_type instead.
        
    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes. 
        Options (GRAPE) include:
            
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
            DEF is RND
        
        (see PulseGen classes for details)
        For the CRAB the this the guess_pulse_type. 

    init_pulse_params : dict
        Parameters for the initial / guess pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    pulse_scaling : float
        Linear scale factor for generated initial / guess pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any initial / guess pulses generated.
        
    ramping_pulse_type : string
        Type of pulse used to modulate the control pulse.
        It's intended use for a ramping modulation, which is often required in 
        experimental setups.
        This is only currently implemented in CRAB.
        GAUSSIAN_EDGE was added for this purpose.
        
    ramping_pulse_params : dict
        Parameters for the ramping pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created
    
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
    opt : Optimizer    
        Instance of an Optimizer, through which the
        Config, Dynamics, PulseGen, and TerminationConditions objects
        can be accessed as attributes.
        The PropagatorComputer, FidelityComputer and TimeslotComputer objects
        can be accessed as attributes of the Dynamics object, e.g. optimizer.dynamics.fid_computer
        The optimisation can be run through the optimizer.run_optimization
    
    """

    # check parameters
    if not isinstance(drift, Qobj):
        if not isinstance(drift, (list, tuple)):
            raise TypeError("drift should be a Qobj or a list of Qobj")
        else:
            for d in drift:
                if not isinstance(d, Qobj):
                    raise TypeError("drift should be a Qobj or a list of Qobj")

    if not isinstance(ctrls, (list, tuple)):
        raise TypeError("ctrls should be a list of Qobj")
    else:
        for ctrl in ctrls:
            if not isinstance(ctrl, Qobj):
                raise TypeError("ctrls should be a list of Qobj")

    if not isinstance(initial, Qobj):
        raise TypeError("initial must be a Qobj")

    if not isinstance(target, Qobj):
        raise TypeError("target must be a Qobj")
        
    # Deprecated parameter management
    if not optim_alg is None:
        optim_method = optim_alg
        _param_deprecation(
            "The 'optim_alg' parameter is deprecated. "
            "Use 'optim_method' instead")
            
    if not max_metric_corr is None:
        if isinstance(method_params, dict):
            if not 'max_metric_corr' in method_params:
                 method_params['max_metric_corr'] = max_metric_corr
        else:
            method_params = {'max_metric_corr':max_metric_corr}
        _param_deprecation(
            "The 'max_metric_corr' parameter is deprecated. "
            "Use 'max_metric_corr' in method_params instead")
            
    if not accuracy_factor is None:
        if isinstance(method_params, dict):
            if not 'accuracy_factor' in method_params:
                 method_params['accuracy_factor'] = accuracy_factor
        else:
            method_params = {'accuracy_factor':accuracy_factor}
        _param_deprecation(
            "The 'accuracy_factor' parameter is deprecated. "
            "Use 'accuracy_factor' in method_params instead")
    
    # phase_option
    if not phase_option is None:
        if isinstance(fid_params, dict):
            if not 'phase_option' in fid_params:
                 fid_params['phase_option'] = phase_option
        else:
            fid_params = {'phase_option':phase_option}
        _param_deprecation(
            "The 'phase_option' parameter is deprecated. "
            "Use 'phase_option' in fid_params instead")
            
    # fid_err_scale_factor
    if not fid_err_scale_factor is None:
        if isinstance(fid_params, dict):
            if not 'fid_err_scale_factor' in fid_params:
                 fid_params['scale_factor'] = fid_err_scale_factor
        else:
            fid_params = {'scale_factor':fid_err_scale_factor}
        _param_deprecation(
            "The 'fid_err_scale_factor' parameter is deprecated. "
            "Use 'scale_factor' in fid_params instead")
            
    # amp_update_mode
    if not amp_update_mode is None:
        amp_update_mode_up = _upper_safe(amp_update_mode)
        if amp_update_mode_up == 'ALL':
            tslot_type = 'UPDATE_ALL'
        else:
            tslot_type = amp_update_mode
        _param_deprecation(
            "The 'amp_update_mode' parameter is deprecated. "
            "Use 'tslot_type' instead")
            
    # set algorithm defaults
    alg_up = _upper_safe(alg)
    if alg is None:
        raise errors.UsageError(
            "Optimisation algorithm must be specified through 'alg' parameter")
    elif alg_up == 'GRAPE':
        if optim_method is None or optim_method.upper() == 'DEF':
            optim_method = 'FMIN_L_BFGS_B'
        if init_pulse_type is None or init_pulse_type.upper() == 'DEF':
            init_pulse_type = 'RND'

    else:
        raise errors.UsageError(
            "No option for pulse optimisation algorithm alg={}".format(alg))

    cfg = optimconfig.OptimConfig()
    cfg.optim_method = optim_method
    cfg.dyn_type = dyn_type
    cfg.prop_type = prop_type
    cfg.fid_type = fid_type
    cfg.init_pulse_type = init_pulse_type

    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    cfg.log_level = log_level

    # Create the Dynamics instance
    if dyn_type == 'GEN_MAT' or dyn_type is None or dyn_type == '':
        dyn = DynamicsGenMat(cfg)
    elif dyn_type == 'UNIT':
        dyn = DynamicsUnitary(cfg)
    elif dyn_type == 'SYMPL':
        dyn = DynamicsSymplectic(cfg)
    else:
        raise errors.UsageError("No option for dyn_type: " + dyn_type)
    dyn.apply_params(dyn_params)
    
    # Create the PropagatorComputer instance
    # The default will be typically be the best option
    if prop_type == 'DEF' or prop_type is None or prop_type == '':
        # Do nothing use the default for the Dynamics
        pass

    # Create the FidelityComputer instance
    # The default will be typically be the best option
    # Note: the FidCompTraceDiffApprox is a subclass of FidCompTraceDiff
    # so need to check this type first
    fid_type_up = _upper_safe(fid_type)
    if fid_type_up == 'DEF' or fid_type_up is None or fid_type_up == '':
        # None given, use the default for the Dynamics
        pass
    elif fid_type_up == 'TDAPPROX':
        if not isinstance(dyn.fid_computer, fidcomp.FidCompTraceDiffApprox):
            dyn.fid_computer = fidcomp.FidCompTraceDiffApprox(dyn)
    elif fid_type_up == 'TRACEDIFF':
        if not isinstance(dyn.fid_computer, fidcomp.FidCompTraceDiff):
            dyn.fid_computer = fidcomp.FidCompTraceDiff(dyn)
    elif fid_type_up == 'UNIT':
        if not isinstance(dyn.fid_computer, fidcomp.FidCompUnitary):
            dyn.fid_computer = fidcomp.FidCompUnitary(dyn)
    else:
        raise errors.UsageError("No option for fid_type: " + fid_type)
    dyn.fid_computer.apply_params(fid_params)
    
    # Currently the only working option for tslot computer is 
    # TSlotCompUpdateAll.
    # so just apply the parameters
    dyn.tslot_computer.apply_params(tslot_params)    

    # Create the Optimiser instance
    optim_method_up = _upper_safe(optim_method)
    if optim_method is None or optim_method_up == '':
        raise errors.UsageError("Optimisation method must be specified "
                                "via 'optim_method' parameter")
    elif optim_method_up == 'FMIN_BFGS':
        optim = optimizer.OptimizerBFGS(cfg, dyn)
    elif optim_method_up == 'LBFGSB' or optim_method_up == 'FMIN_L_BFGS_B':
        optim = optimizer.OptimizerLBFGSB(cfg, dyn)
    elif optim_method_up == 'FMIN':
        if alg_up == 'CRAB':
            optim = optimizer.OptimizerCrabFmin(cfg, dyn)
        else:
            raise errors.UsageError(
                "Invalid optim_method '{}' for '{}' algorthim".format(
                                    optim_method, alg))
    else:
        # Assume that the optim_method is a valid
        #scipy.optimize.minimize method
        # Choose an optimiser based on the algorithm
        if alg_up == 'CRAB':
            optim = optimizer.OptimizerCrab(cfg, dyn)
        else:
            optim = optimizer.Optimizer(cfg, dyn)
    
    optim.alg = alg
    optim.method = optim_method
    optim.amp_lbound = amp_lbound
    optim.amp_ubound = amp_ubound
    optim.apply_params(optim_params)
    
    # Create the TerminationConditions instance
    tc = termcond.TerminationConditions()
    tc.fid_err_targ = fid_err_targ
    tc.min_gradient_norm = min_grad
    tc.max_iterations = max_iter
    tc.max_wall_time = max_wall_time
    optim.termination_conditions = tc
    
    
    optim.apply_method_params(method_params)

    if gen_stats:
        # Create a stats object
        # Note that stats object is optional
        # if the Dynamics and Optimizer stats attribute is not set
        # then no stats will be collected, which could improve performance
        if amp_update_mode == 'DYNAMIC':
            sts = stats.StatsDynTsUpdate()
        else:
            sts = stats.Stats()

        dyn.stats = sts
        optim.stats = sts

    # Configure the dynamics
    dyn.drift_dyn_gen = drift
    dyn.ctrl_dyn_gen = ctrls
    dyn.initial = initial
    dyn.target = target
    if tau is None:
        # Check that parameters have been supplied to generate the
        # timeslot durations
        try:
            evo_time / num_tslots
        except:
            raise errors.UsageError(
                "Either the timeslot durations should be supplied as an "
                "array 'tau' or the number of timeslots 'num_tslots' "
                "and the evolution time 'evo_time' must be given.")

        dyn.num_tslots = num_tslots
        dyn.evo_time = evo_time
    else:
        dyn.tau = tau

    # this function is called, so that the num_ctrls attribute will be set
    n_ctrls = dyn.num_ctrls

    ramping_pgen = None
    if ramping_pulse_type:
        ramping_pgen = pulsegen.create_pulse_gen(
                            pulse_type=ramping_pulse_type, dyn=dyn, 
                            pulse_params=ramping_pulse_params)
    if alg_up == 'CRAB':
        # Create a pulse generator for each ctrl
        crab_pulse_params = None
        num_coeffs = None
        init_coeff_scaling = None
        if isinstance(alg_params, dict):
            num_coeffs = alg_params.get('num_coeffs')
            init_coeff_scaling = alg_params.get('init_coeff_scaling')
            if 'crab_pulse_params' in alg_params:
                crab_pulse_params = alg_params.get('crab_pulse_params')
            
        guess_pulse_type = init_pulse_type
        if guess_pulse_type:
            guess_pulse_action = None
            guess_pgen = pulsegen.create_pulse_gen(
                                pulse_type=guess_pulse_type, dyn=dyn)
            guess_pgen.scaling = pulse_scaling
            guess_pgen.offset = pulse_offset
            if init_pulse_params is not None:
                guess_pgen.apply_params(init_pulse_params)
                guess_pulse_action = init_pulse_params.get('pulse_action')

        optim.pulse_generator = []
        for j in range(n_ctrls):
            crab_pgen = pulsegen.PulseGenCrabFourier(
                                dyn=dyn, num_coeffs=num_coeffs)
            if init_coeff_scaling is not None:
                crab_pgen.scaling = init_coeff_scaling
            if isinstance(crab_pulse_params, dict):
                crab_pgen.apply_params(crab_pulse_params)
                
            lb = None
            if amp_lbound:
                if isinstance(amp_lbound, list):
                    try:
                        lb = amp_lbound[j]
                    except:
                        lb = amp_lbound[-1]
                else:
                    lb = amp_lbound
            ub = None
            if amp_ubound:
                if isinstance(amp_ubound, list):
                    try:
                        ub = amp_ubound[j]
                    except:
                        ub = amp_ubound[-1]
                else:
                    ub = amp_ubound
            crab_pgen.lbound = lb
            crab_pgen.ubound = ub
            
            if guess_pulse_type:
                guess_pgen.lbound = lb
                guess_pgen.ubound = ub
                crab_pgen.guess_pulse = guess_pgen.gen_pulse()
                if guess_pulse_action:
                    crab_pgen.guess_pulse_action = guess_pulse_action
                
            if ramping_pgen:
                crab_pgen.ramping_pulse = ramping_pgen.gen_pulse()

            optim.pulse_generator.append(crab_pgen)
        #This is just for the debug message now
        pgen = optim.pulse_generator[0]
            
    else:
        # Create a pulse generator of the type specified
        pgen = pulsegen.create_pulse_gen(pulse_type=init_pulse_type, dyn=dyn,
                                        pulse_params=init_pulse_params)
        pgen.scaling = pulse_scaling
        pgen.offset = pulse_offset
        pgen.lbound = amp_lbound
        pgen.ubound = amp_ubound

        optim.pulse_generator = pgen

    if log_level <= logging.DEBUG:
        logger.debug(
            "Optimisation config summary...\n"
            "  object classes:\n"
            "    optimizer: " + optim.__class__.__name__ +
            "\n    dynamics: " + dyn.__class__.__name__ +
            "\n    tslotcomp: " + dyn.tslot_computer.__class__.__name__ +
            "\n    fidcomp: " + dyn.fid_computer.__class__.__name__ +
            "\n    propcomp: " + dyn.prop_computer.__class__.__name__ +
            "\n    pulsegen: " + pgen.__class__.__name__)

    return optim

def optimize_pulse(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        alg='GRAPE', alg_params=None,
        optim_params=None, optim_method='DEF', method_params=None,
        optim_alg=None, max_metric_corr=None, accuracy_factor=None,
        dyn_type='GEN_MAT', dyn_params=None,
        prop_type='DEF', prop_params=None,
        fid_type='DEF', fid_params=None,
        phase_option=None, fid_err_scale_factor=None,
        tslot_type='DEF', tslot_params=None,
        amp_update_mode=None,
        init_pulse_type='DEF', init_pulse_params=None,
        pulse_scaling=1.0, pulse_offset=0.0,
        ramping_pulse_type=None, ramping_pulse_params=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False):
    
    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)
        
    # The parameters types are checked in create_pulse_optimizer
    # so no need to do so here
    # However, the deprecation management is repeated here
    # so that the stack level is correct
    if not optim_alg is None:
        optim_method = optim_alg
        _param_deprecation(
            "The 'optim_alg' parameter is deprecated. "
            "Use 'optim_method' instead")
            
    if not max_metric_corr is None:
        if isinstance(method_params, dict):
            if not 'max_metric_corr' in method_params:
                 method_params['max_metric_corr'] = max_metric_corr
        else:
            method_params = {'max_metric_corr':max_metric_corr}
        _param_deprecation(
            "The 'max_metric_corr' parameter is deprecated. "
            "Use 'max_metric_corr' in method_params instead")
            
    if not accuracy_factor is None:
        if isinstance(method_params, dict):
            if not 'accuracy_factor' in method_params:
                 method_params['accuracy_factor'] = accuracy_factor
        else:
            method_params = {'accuracy_factor':accuracy_factor}
        _param_deprecation(
            "The 'accuracy_factor' parameter is deprecated. "
            "Use 'accuracy_factor' in method_params instead")
    
    # phase_option
    if not phase_option is None:
        if isinstance(fid_params, dict):
            if not 'phase_option' in fid_params:
                 fid_params['phase_option'] = phase_option
        else:
            fid_params = {'phase_option':phase_option}
        _param_deprecation(
            "The 'phase_option' parameter is deprecated. "
            "Use 'phase_option' in fid_params instead")
            
    # fid_err_scale_factor
    if not fid_err_scale_factor is None:
        if isinstance(fid_params, dict):
            if not 'fid_err_scale_factor' in fid_params:
                 fid_params['scale_factor'] = fid_err_scale_factor
        else:
            fid_params = {'scale_factor':fid_err_scale_factor}
        _param_deprecation(
            "The 'fid_err_scale_factor' parameter is deprecated. "
            "Use 'scale_factor' in fid_params instead")
            
    # amp_update_mode
    if not amp_update_mode is None:
        amp_update_mode_up = _upper_safe(amp_update_mode)
        if amp_update_mode_up == 'ALL':
            tslot_type = 'UPDATE_ALL'
        else:
            tslot_type = amp_update_mode
        _param_deprecation(
            "The 'amp_update_mode' parameter is deprecated. "
            "Use 'tslot_type' instead")

    optim = create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ, min_grad=min_grad,
        max_iter=max_iter, max_wall_time=max_wall_time,
        alg=alg, alg_params=alg_params, optim_params=optim_params,
        optim_method=optim_method, method_params=method_params,
        dyn_type=dyn_type, dyn_params=dyn_params, 
        prop_type=prop_type, prop_params=prop_params,
        fid_type=fid_type, fid_params=fid_params,
        init_pulse_type=init_pulse_type, init_pulse_params=init_pulse_params,
        pulse_scaling=pulse_scaling, pulse_offset=pulse_offset,
        ramping_pulse_type=ramping_pulse_type, 
        ramping_pulse_params=ramping_pulse_params,
        log_level=log_level, gen_stats=gen_stats)

    dyn = optim.dynamics

    dyn.init_timeslots()
    # Generate initial pulses for each control
    init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])
    
    if alg == 'CRAB':
        for j in range(dyn.num_ctrls):
            pgen = optim.pulse_generator[j]
            pgen.init_pulse()
            init_amps[:, j] = pgen.gen_pulse()
    else:
        pgen = optim.pulse_generator
        for j in range(dyn.num_ctrls):
            init_amps[:, j] = pgen.gen_pulse()
        
    # Initialise the starting amplitudes
    dyn.initialize_controls(init_amps)
    
    if log_level <= logging.INFO:
        msg = "System configuration:\n"
        dg_name = "dynamics generator"
        if dyn_type == 'UNIT':
            dg_name = "Hamiltonian"
        if dyn.time_depend_drift:
            msg += "Initial drift {}:\n".format(dg_name)
            msg += str(dyn.drift_dyn_gen[0])
        else:
            msg += "Drift {}:\n".format(dg_name)
            msg += str(dyn.drift_dyn_gen)
        for j in range(dyn.num_ctrls):
            msg += "\nControl {} {}:\n".format(j+1, dg_name)
            msg += str(dyn.ctrl_dyn_gen[j])
        msg += "\nInitial state / operator:\n"
        msg += str(dyn.initial)
        msg += "\nTarget state / operator:\n"
        msg += str(dyn.target)
        logger.info(msg)

    if out_file_ext is not None:
        # Save initial amplitudes to a text file
        pulsefile = "ctrl_amps_initial_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Initial amplitudes output to file: " + pulsefile)

    # Start the optimisation
    result = optim.run_optimization()

    if out_file_ext is not None:
        # Save final amplitudes to a text file
        pulsefile = "ctrl_amps_final_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Final amplitudes output to file: " + pulsefile)

    return result

def optimize_pulse_unitary(
        H_d, H_c, U_0, U_targ,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10,
        max_iter=500, max_wall_time=180,
        alg='GRAPE', alg_params=None,
        optim_params=None, optim_method='DEF', method_params=None,
        optim_alg=None, max_metric_corr=None, accuracy_factor=None,
        phase_option='PSU', 
        dyn_params=None, prop_params=None, fid_params=None,
        tslot_type='DEF', tslot_params=None,
        amp_update_mode=None,
        init_pulse_type='DEF', init_pulse_params=None,
        pulse_scaling=1.0, pulse_offset=0.0,
        ramping_pulse_type=None, ramping_pulse_params=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False):
      
    return optimize_pulse(
            drift=H_d, ctrls=H_c, initial=U_0, target=U_targ,
            num_tslots=num_tslots, evo_time=evo_time, tau=tau,
            amp_lbound=amp_lbound, amp_ubound=amp_ubound,
            fid_err_targ=fid_err_targ, min_grad=min_grad,
            max_iter=max_iter, max_wall_time=max_wall_time,
            alg=alg, alg_params=alg_params, optim_params=optim_params,
            optim_method=optim_method, method_params=method_params,
            dyn_type='UNIT', dyn_params=dyn_params,
            prop_params=prop_params, fid_params=fid_params,
            init_pulse_type=init_pulse_type, init_pulse_params=init_pulse_params,
            pulse_scaling=pulse_scaling, pulse_offset=pulse_offset,
            ramping_pulse_type=ramping_pulse_type, 
            ramping_pulse_params=ramping_pulse_params,
            log_level=log_level, out_file_ext=out_file_ext,
            gen_stats=gen_stats)


class OptPulseProcessor(Processor):
    """
    A processor that uses
    :obj:`qutip.control.optimize_pulse_unitary`
    to find optimized pulses for a given quantum circuit.
    The processor can simulate the evolution under the given
    control pulses using :func:`qutip.mesolve`.
    (For attributes documentation, please
    refer to the parent class :class:`.Processor`)

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    drift: `:class:`qutip.Qobj`
        The drift Hamiltonian. The size must match the whole quantum system.

    dims: list
        The dimension of each component system.
        Default value is a qubit system of ``dim=[2,2,2,...,2]``

    **params:
        - t1 : float or list, optional
            Characterize the amplitude damping for each qubit.
            A list of size `num_qubits` or a float for all qubits.
        - t2 : float or list, optional
            Characterize the total dephasing for each qubit.
            A list of size `num_qubits` or a float for all qubits.
    """

    def __init__(self, num_qubits=None, drift=None, dims=None, **params):
        super(OptPulseProcessor, self).__init__(
            num_qubits, dims=dims, **params
        )
        if drift is not None:
            self.add_drift(drift, list(range(self.num_qubits)))
        self.spline_kind = "step_func"

    def load_circuit(
        self,
        qc,
        min_fid_err=np.inf,
        merge_gates=True,
        setting_args=None,
        verbose=False,
        **kwargs
    ):
        """
        Find the pulses realizing a given :class:`.Circuit` using
        :func:`qutip.control.optimize_pulse_unitary`. Further parameter for
        for :func:`qutip.control.optimize_pulse_unitary` needs to be given as
        keyword arguments. By default, it first merge all the gates
        into one unitary and then find the control pulses for it.
        It can be turned off and one can set different parameters
        for different gates. See examples for details.

        Examples
        --------
        Same parameter for all the gates

        >>> from qutip_qip.circuit import QubitCircuit
        >>> from qutip_qip.device import OptPulseProcessor
        >>> qc = QubitCircuit(1)
        >>> qc.add_gate("SNOT", 0)
        >>> num_tslots = 10
        >>> evo_time = 10
        >>> processor = OptPulseProcessor(1, drift=sigmaz())
        >>> processor.add_control(sigmax())
        >>> # num_tslots and evo_time are two keyword arguments
        >>> tlist, coeffs = processor.load_circuit(\
                qc, num_tslots=num_tslots, evo_time=evo_time)

        Different parameters for different gates

        >>> from qutip_qip.circuit import QubitCircuit
        >>> from qutip_qip.device import OptPulseProcessor
        >>> qc = QubitCircuit(2)
        >>> qc.add_gate("SNOT", 0)
        >>> qc.add_gate("SWAP", targets=[0, 1])
        >>> qc.add_gate('CNOT', controls=1, targets=[0])
        >>> processor = OptPulseProcessor(2, drift=tensor([sigmaz()]*2))
        >>> processor.add_control(sigmax(), cyclic_permutation=True)
        >>> processor.add_control(sigmay(), cyclic_permutation=True)
        >>> processor.add_control(tensor([sigmay(), sigmay()]))
        >>> setting_args = {"SNOT": {"num_tslots": 10, "evo_time": 1},\
                        "SWAP": {"num_tslots": 30, "evo_time": 3},\
                        "CNOT": {"num_tslots": 30, "evo_time": 3}}
        >>> tlist, coeffs = processor.load_circuit(\
                qc, setting_args=setting_args, merge_gates=False)

        Parameters
        ----------
        qc : :class:`.QubitCircuit` or list of Qobj
            The quantum circuit to be translated.

        min_fid_err: float, optional
            The minimal fidelity tolerance, if the fidelity error of any
            gate decomposition is higher, a warning will be given.
            Default is infinite.

        merge_gates: boolean, optimal
            If True, merge all gate/Qobj into one Qobj and then
            find the optimal pulses for this unitary matrix. If False,
            find the optimal pulses for each gate/Qobj.

        setting_args: dict, optional
            Only considered if merge_gates is False.
            It is a dictionary containing keyword arguments
            for different gates.

        verbose: boolean, optional
            If true, the information for each decomposed gate
            will be shown. Default is False.

        **kwargs
            keyword arguments for
            :func:``qutip.control.optimize_pulse_unitary``

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape ``(len(ctrls), len(tlist)-1)``. Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.

        Notes
        -----
        ``len(tlist)-1=coeffs.shape[1]`` since tlist gives
        the beginning and the end of the pulses
        """
        if setting_args is None:
            setting_args = {}
        if isinstance(qc, QubitCircuit):
            props = qc.propagators()
            gates = [g.name for g in qc.gates]
        elif isinstance(qc, Iterable):
            props = qc
            gates = None  # using list of Qobj, no gates name
        else:
            raise ValueError(
                "qc should be a " "QubitCircuit or a list of Qobj"
            )
        if merge_gates:  # merge all gates/Qobj into one Qobj
            props = [gate_sequence_product(props)]
            gates = None

        time_record = []  # a list for all the gates
        coeff_record = []
        last_time = 0.0  # used in concatenation of tlist
        for prop_ind, U_targ in enumerate(props):
            U_0 = identity(U_targ.dims[0])

            # If qc is a QubitCircuit and setting_args is not empty,
            # we update the kwargs for each gate.
            # keyword arguments in setting_arg have priority
            if gates is not None and setting_args:
                kwargs.update(setting_args[gates[prop_ind]])

            control_labels = self.model.get_control_labels()
            full_ctrls_hams = []
            for label in control_labels:
                qobj, targets = self.model.get_control(label)
                full_ctrls_hams.append(
                    expand_operator(qobj, dims=self.dims, targets=targets)
                )

            full_drift_ham = sum(
                [
                    expand_operator(qobj, dims=self.dims, targets=targets)
                    for (qobj, targets) in self.model.get_all_drift()
                ],
                Qobj(
                    np.zeros(full_ctrls_hams[0].shape),
                    dims=[self.dims, self.dims],
                ),
            )

            result = cpo.optimize_pulse_unitary(
                full_drift_ham, full_ctrls_hams, U_0, U_targ, **kwargs
            )

            if result.fid_err > min_fid_err:
                warnings.warn(
                    "The fidelity error of gate {} is higher "
                    "than required limit. Use verbose=True to see"
                    "the more detailed information.".format(prop_ind)
                )

            time_record.append(result.time[1:] + last_time)
            last_time += result.time[-1]
            coeff_record.append(result.final_amps.T)

            if verbose:
                print("********** Gate {} **********".format(prop_ind))
                print("Final fidelity error {}".format(result.fid_err))
                print(
                    "Final gradient normal {}".format(result.grad_norm_final)
                )
                print("Terminated due to {}".format(result.termination_reason))
                print("Number of iterations {}".format(result.num_iter))

        tlist = np.hstack([[0.0]] + time_record)
        for i in range(len(self.pulses)):
            self.pulses[i].tlist = tlist
        coeffs = np.vstack([np.hstack(coeff_record)])

        coeffs = {label: coeff for label, coeff in zip(control_labels, coeffs)}
        self.set_coeffs(coeffs)
        self.set_tlist(tlist)
        return tlist, coeffs