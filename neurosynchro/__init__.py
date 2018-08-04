# -*- mode: python; coding: utf-8 -*-
# Copyright 2017-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Train and use artificial neural network approximations ("regressions") of
synchrotron radiative transfer coefficients as a function of various physical
input parameters.

The meat of the neural network code is in the ``impl`` module to avoid
importing Keras unless needed.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
AbsLogMapping
DirectMapping
DomainRange
LogMapping
LogitMapping
Mapping
NegLogMapping
NinthRootMapping
SampleData
SignMapping
basic_load
detrivialize_stokes_basis
mapping_from_dict
mapping_from_samples
'''.split()

from collections import OrderedDict
from six.moves import range
import numpy as np
import pandas as pd
from pwkit.io import Path


class Mapping(object):
    """An abstract base class for parameter transformations.

    Mapping classes are used to translate the physical parameters fed into
    *neurosynchro* into normalized values that are easier to work with
    numerically. You will not normally need to use them directly.

    """
    desc = None # set by subclasses
    trainer = None

    phys_bounds_mode = 'empirical'
    "Blah blah"

    normalization_mode = 'gaussian'
    out_of_sample = 'ignore'

    def __init__(self, name):
        self.name = name


    @classmethod
    def from_info_and_samples(cls, info, phys_samples):
        """Create a new :class:`Mapping` from a dictionary of information and a set of samples.

        **Call signature**

        *info*
          A dictionary of attributes, passed into :meth:`Mapping.from_dict`.
        *phys_samples*
          A 1D Numpy array of samples of this parameter, in no particular order.
        Return value
          A new instance of :class:`Mapping` (or one of its subclasses) with initialized
          bounds parameters.

        """
        inst = cls.from_dict(info, load_bounds=False)

        valid = np.isfinite(phys_samples) & inst._is_valid(phys_samples)
        n_rej = phys_samples.size - valid.sum()
        print('%s: rejecting %d samples out of %d' % (inst.name, n_rej, phys_samples.size))
        phys_samples = phys_samples[valid]
        if phys_samples.size < 3:
            raise Exception('not enough valid samples for %s' % inst.name)

        if inst.phys_bounds_mode == 'empirical':
            inst.p_min = phys_samples.min()
            inst.p_max = phys_samples.max()
        elif inst.phys_bounds_mode == 'theta':
            inst.p_min = 0.
            inst.p_max = 0.5 * np.pi
        else:
            raise ValueError('unrecognized phys_bounds_mode value %r for %s' %
                             (inst.phys_bounds_mode, inst.name))

        # Pluggable "transform"
        transformed = inst._to_xform(phys_samples)

        if inst.normalization_mode == 'gaussian':
            inst.x_mean = transformed.mean()
            inst.x_std = transformed.std()
        elif inst.normalization_mode == 'unit_interval':
            # Maps the physical values to the unit interval [0, 1].
            inst.x_mean = transformed.min()
            inst.x_std = transformed.max() - inst.x_mean
            if inst.x_std == 0:
                inst.x_std = 1.
        else:
            raise ValueError('unrecognized normalization_mode value %r for %s' %
                             (inst.normalization_mode, inst.name))

        # Normalize
        normed = (transformed - inst.x_mean) / inst.x_std
        inst.n_min = normed.min()
        inst.n_max = normed.max()

        return inst


    def __repr__(self):
        return '<Mapping %s %s mean=%r sd=%r>' % (self.name, self.desc, self.x_mean, self.x_std)


    def phys_to_norm(self, phys):
        """Map "physical" parameters to normalized values

        **Argument**

        *phys*
          An array of "physical" input values (see :ref:`transformations`).

        **Return values**

        This method returns a tuple ``(normalized, oos)``.

        *normalized*
          The normalized versions of the input data.
        *oos*
          An array of booleans of the same shape as the input data. True
          values indicate inputs that were out of the sample that was used to
          define the mapping.

        """
        # note: using prefix ~ instead of np.logical_not fails for scalars
        oos = np.logical_not((phys >= self.p_min) & (phys <= self.p_max)) # catches NaNs
        any_oos = np.any(oos)

        if any_oos:
            if self.out_of_sample == 'ignore':
                pass
            elif self.out_of_sample == 'clip':
                phys = np.clip(phys, self.p_min, self.p_max)
            elif self.out_of_sample == 'nan':
                phys = phys.copy()
                phys[oos] = np.nan
            else:
                raise Exception('unrecognized out-of-sample behavior %r' % self.out_of_sample)

        return (self._to_xform(phys) - self.x_mean) / self.x_std, any_oos


    def norm_to_phys(self, norm):
        """Map "normalized" parameters to "physical" values

        **Argument**

        *norm*
          An array of "normalized" input values (see :ref:`transformations`).

        **Return values**

        This method returns a tuple ``(phys, oos)``.

        *phys*
          The physical versions of the input data.
        *oos*
          An array of booleans of the same shape as the input data. True
          values indicate inputs that were out of the sample that was used to
          define the mapping.

        """
        oos = np.logical_not((norm >= self.n_min) & (norm <= self.n_max)) # catches NaNs
        any_oos = np.any(oos)

        if any_oos:
            if self.out_of_sample == 'ignore':
                pass
            elif self.out_of_sample == 'clip':
                norm = np.clip(norm, self.n_min, self.n_max)
            elif self.out_of_sample == 'nan':
                norm = norm.copy()
                norm[oos] = np.nan
            else:
                raise Exception('unrecognized out-of-sample behavior %r' % self.out_of_sample)

        return self._from_xform(norm * self.x_std + self.x_mean), any_oos


    def to_dict(self):
        """Serialize this :class:`Mapping` into an ordered dictionary."""
        d = OrderedDict()
        d['name'] = self.name
        d['maptype'] = self.desc

        if self.phys_bounds_mode is not None:
            d['phys_bounds_mode'] = self.phys_bounds_mode
        if self.normalization_mode is not None:
            d['normalization_mode'] = self.normalization_mode
        if self.trainer is not None:
            d['trainer'] = self.trainer
        if self.out_of_sample is not None:
            d['out_of_sample'] = self.out_of_sample

        d['x_mean'] = self.x_mean
        d['x_std'] = self.x_std
        d['phys_min'] = self.p_min
        d['phys_max'] = self.p_max
        d['norm_min'] = self.n_min
        d['norm_max'] = self.n_max
        return d


    @classmethod
    def from_dict(cls, info, load_bounds=True):
        """Deserialize an ordered dictionary into a new :class:`Mapping` instance.

        **Call signature**

        *info*
          A dictionary of parameters, as generated by :meth:`Mapping.to_dict`.
        *load_bounds* (default: :const:`True`)
          If true, deserialize bounds information such as the maximum and minimum
          observed physical values. If :const:`False`, these are left uninitialized.
        Return value
          A new :class:`Mapping` instance.

        """
        if str(info['maptype']) != cls.desc:
            raise ValueError('info is for maptype %s but this class is %s' % (info['maptype'], cls.desc))

        inst = cls(str(info['name']))
        if 'phys_bounds_mode' in info:
            inst.phys_bounds_mode = info['phys_bounds_mode']
        if 'normalization_mode' in info:
            inst.normalization_mode = info['normalization_mode']
        if 'trainer' in info:
            inst.trainer = info['trainer']
        if 'out_of_sample' in info:
            inst.out_of_sample = info['out_of_sample']

        if load_bounds:
            inst.x_mean = float(info['x_mean'])
            inst.x_std = float(info['x_std'])
            inst.p_min = float(info['phys_min'])
            inst.p_max = float(info['phys_max'])
            inst.n_min = float(info['norm_min'])
            inst.n_max = float(info['norm_max'])

        return inst


class AbsLogMapping(Mapping):
    desc = 'abs_log'

    def _to_xform(self, p):
        return np.log10(np.abs(p))

    def _from_xform(self, x):
        return 10**x # XXX not invertible!

    def _is_valid(self, p):
        return p != 0


class DirectMapping(Mapping):
    desc = 'direct'

    def _to_xform(self, p):
        return p

    def _from_xform(self, x):
        return x

    def _is_valid(self, p):
        return np.ones(p.shape, dtype=np.bool)


class LogMapping(Mapping):
    desc = 'log'

    def _to_xform(self, p):
        return np.log10(p)

    def _from_xform(self, x):
        return 10**x

    def _is_valid(self, p):
        return (p > 0)


class LogitMapping(Mapping):
    desc = 'logit'

    def _to_xform(self, p):
        return np.log(p / (1. - p))

    def _from_xform(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def _is_valid(self, p):
        # Infinities are hard to deal with so we don't allow p = 0 or p = 1.
        return (p > 0) & (p < 1)


class NegLogMapping(Mapping):
    desc = 'neg_log'

    def _to_xform(self, p):
        return np.log10(-p)

    def _from_xform(self, x):
        return -(10**x)

    def _is_valid(self, p):
        return (p < 0)


class NinthRootMapping(Mapping):
    desc = 'ninth_root'

    def _to_xform(self, p):
        return np.cbrt(np.cbrt(p))

    def _from_xform(self, x):
        return x**9

    def _is_valid(self, p):
        return np.ones(p.shape, dtype=np.bool)


class SignMapping(Mapping):
    desc = 'sign'

    def _to_xform(self, p):
        return np.sign(p)

    def _from_xform(self, x):
        return np.sign(x) # XXX not reversible!

    def _is_valid(self, p):
        return (p != 0.)


_mappings = {
    'abs_log': AbsLogMapping,
    'direct': DirectMapping,
    'log': LogMapping,
    'logit': LogitMapping,
    'neg_log': NegLogMapping,
    'ninth_root': NinthRootMapping,
    'sign': SignMapping,
}


class Passthrough(object):
    def __init__(self, d):
        self.d = d

    def to_dict(self):
        return self.d


def mapping_from_info_and_samples(info, phys_samples):
    """Create a :class:`Mapping` subclass from configuration data and sample data.

    **Call signature**

    *info*
      A dictionary of configuration info, loaded from ``nn_config.toml`` or
      created by :meth:`Mapping.to_dict`.
    *phys_samples*
      A 1D array of training data used to initialize the bounds of this mapping.
    Return value
      A new :class:`Mapping` instance. The particular subclass used depends on the
      ``maptype`` setting, as documented in the :ref:`map-types`.

    """
    cls = _mappings[info['maptype']]
    return cls.from_info_and_samples(info, phys_samples)

def mapping_from_dict(info):
    """Create a :class:`Mapping` subclass from configuration data.

    **Call signature**

    *info*
      A dictionary of configuration info, loaded from ``nn_config.toml`` or
      created by :meth:`Mapping.to_dict`.
    Return value
      A new :class:`Mapping` instance. The particular subclass used depends on the
      ``maptype`` setting, as documented in the :ref:`map-types`.

    Unlike :func:`mapping_from_info_and_samples`, the bounds data are not initialized
    if they are not already defined in the configuration dictionary.

    """
    maptype = str(info['maptype'])
    cls = _mappings[maptype]
    return cls.from_dict(info)


def basic_load(datadir, drop_metadata=True):
    """Load a directory of textual tables (such as training set data).

    **Call signature**

    *datadir*
      A path to a directory of textual tables; format described below.
    *drop_metadata* (default ``True``)
      If true, columns marked as metadata will be dropped from the returned
      table.
    Return value
      A :class:`pandas.DataFrame` of data, concatenating all of the input tables.

    The data format is documented in :ref:`make-training-set`. Briefly, each
    file in *datadir* whose name ends with ``.txt`` will be loaded as a table
    using :func:`pandas.read_table`. The recommended format is tab-separated
    values with a single header row. Column names should end in type
    identifiers such as ``(lin)`` to identify their roles, although this
    function ignores this information except to drop columns whose names end
    in ``(meta)`` if so directed.

    """
    datadir = Path(datadir)
    chunks = []

    for item in datadir.glob('*.txt'):
        chunks.append(pd.read_table(str(item)))

    data = pd.concat(chunks, ignore_index=True)

    if drop_metadata:
        # Drop `foo(mtea)` columns
        for col in data.columns:
            if col.endswith('(meta)'):
                del data[col]

    return data


class DomainRange(object):
    n_params = None
    n_results = None
    pmaps = None
    rmaps = None


    @classmethod
    def from_info_and_samples(cls, info, df):
        inst = cls()
        inst.n_params = len(info['params'])
        inst.n_results = len(info['results'])
        inst.pmaps = []
        inst.rmaps = []

        for i, pinfo in enumerate(info['params']):
            if not df.columns[i].startswith(pinfo['name'] + '('):
                raise Exception('alignment error: expect data column %d to be %r; got %r' % (i, pinfo['name'], df.columns[i]))
            inst.pmaps.append(mapping_from_info_and_samples(pinfo, df[df.columns[i]].values))

        n = inst.n_params

        for i, rinfo in enumerate(info['results']):
            if not df.columns[n+i].startswith(rinfo['name'] + '(res)'):
                raise Exception('alignment error: expect data column %d to be %r; got %r' % (n + i, rinfo['name'], df.columns[n + i]))
            inst.rmaps.append(mapping_from_info_and_samples(rinfo, df[df.columns[n+i]].values))

        return inst


    @classmethod
    def from_serialized(cls, config_path, result_to_extract=None):
        """`result_to_extract` is a total lazy hack for the training tool."""
        import pytoml

        with Path(config_path).open('rt') as f:
            info = pytoml.load(f)

        inst = cls()
        inst.pmaps = []
        inst.rmaps = []
        extracted_info = None

        for subinfo in info['params']:
            inst.pmaps.append(mapping_from_dict(subinfo))

        for i, subinfo in enumerate(info['results']):
            if result_to_extract is not None and subinfo['name'] == result_to_extract:
                extracted_info = subinfo
                extracted_info['_index'] = i
            inst.rmaps.append(mapping_from_dict(subinfo))

        inst.n_params = len(inst.pmaps)
        inst.n_results = len(inst.rmaps)

        if result_to_extract is not None:
            return inst, extracted_info
        return inst


    def __repr__(self):
        return '\n'.join(
            ['<%s n_p=%d n_r=%d' % (self.__class__.__name__, self.n_params, self.n_results)] +
            ['  P%d=%r,' % (i, m) for i, m in enumerate(self.pmaps)] +
            ['  R%d=%r,' % (i, m) for i, m in enumerate(self.rmaps)] +
            ['>'])


    def into_info(self, info):
        info['params'] = [m.to_dict() for m in self.pmaps]
        info['results'] = [m.to_dict() for m in self.rmaps]


    def load_and_normalize(self, datadir):
        df = basic_load(datadir)

        for i, pmap in enumerate(self.pmaps):
            if not df.columns[i].startswith(pmap.name + '('):
                raise Exception('alignment error: expect data column %d to be %s(...); got %s' % (i, pmap.name, df.columns[i]))

        n = self.n_params

        for i, rmap in enumerate(self.rmaps):
            if df.columns[n+i] != rmap.name + '(res)':
                raise Exception('alignment error: expect data column %d to be %s(res); got %s' % (n + i, rmap.name, df.columns[n + i]))

        return SampleData(self, df)


class SampleData(object):
    df = None
    domain_range = None
    phys = None
    norm = None
    oos_flags = 0

    def __init__(self, domain_range, df):
        self.df = df
        self.domain_range = domain_range
        self.phys = df.values
        self.norm = np.empty_like(self.phys)

        for i in range(self.domain_range.n_params):
            self.norm[:,i], flag = self.domain_range.pmaps[i].phys_to_norm(self.phys[:,i])
            if flag:
                self.oos_flags |= (1 << i)

        for i in range(self.domain_range.n_results):
            j = i + self.domain_range.n_params
            self.norm[:,j], flag = self.domain_range.rmaps[i].phys_to_norm(self.phys[:,j])
            if flag:
                self.oos_flags |= (1 << j)

    @property
    def phys_params(self):
        return self.phys[:,:self.domain_range.n_params]

    @property
    def phys_results(self):
        return self.phys[:,self.domain_range.n_params:]

    @property
    def norm_params(self):
        return self.norm[:,:self.domain_range.n_params]

    @property
    def norm_results(self):
        return self.norm[:,self.domain_range.n_params:]


def detrivialize_stokes_basis(coeffs, psi):
    """Re-express coefficients in a basis in which the magnetic field can rotate
    on the sky.

    **Arguments**

    *coeffs*
      Radiative transfer coefficients in the Stokes basis where the Stokes U
      axis is aligned with the magnetic field. This is an array of shape ``(S,
      8)`` where *S* is the shape of *psi*. Along the inner axis of the array,
      the coefficients are: ``(j_I, alpha_I, j_Q, alpha_Q, j_V, alpha_V,
      rho_Q, rho_V)``. This is the representation returned by
      :meth:`neurosynchro.impl.PhysicalApproximator.compute_all_nontrivial`.

    *psi*
      The angle(s) between the magnetic field as projected on the sky and some
      invariant Stokes U axis, in radians. XXX: sign convention?

    **Return value**

    An array of radiative transfer coefficients in which the Stokes U terms
    are no longer trivial. The shape is ``(S, 11)``. Along the inner axis of
    the array, the coefficients are: ``(j_I, alpha_I, j_Q, alpha_Q, j_U,
    alpha_U, j_V, alpha_V, rho_Q, rho_U, rho_V)``.

    **Details**

    Synchrotron calculations are generally performed in a basis in which the
    Stokes U axis is aligned with the magnetic field, which means that the
    corresponding radiative transfer coefficients are zero ("trivial"). In
    actual work, however, the magnetic field orientation is not guaranteed to
    be constant along the direction of propagation. If the Stokes U axis is
    held fixed along the integration, the Stokes U coefficients become
    nontrivial. This function transforms coefficients from the former basis
    to the latter.

    See Shcherbakov & Huang (2011MNRAS.410.1052S), equations 50-51. Note
    that the linear polarization axis rotates at twice the rate that psi
    does, because linear polarization is an *orientation* not an *angle*.

    """
    ji = coeffs[...,0]
    ai = coeffs[...,1]
    jq = coeffs[...,2]
    aq = coeffs[...,3]
    jv = coeffs[...,4]
    av = coeffs[...,5]
    rq = coeffs[...,6]
    rv = coeffs[...,7]

    twochi = 2 * (np.pi - psi) # note the sign convention!
    s = np.sin(twochi)
    c = np.cos(twochi)

    xformed = np.empty(coeffs.shape[:-1] + (11,))
    xformed[...,0] = ji # j_I
    xformed[...,1] = ai # alpha_I
    xformed[...,2] = c * jq # j_Q
    xformed[...,3] = c * aq # alpha_Q
    xformed[...,4] = s * jq # j_U
    xformed[...,5] = s * aq # alpha_U
    xformed[...,6] = jv # j_V
    xformed[...,7] = av # alpha_V
    xformed[...,8] = c * rq # rho_Q
    xformed[...,9] = s * rq # rho_U
    xformed[...,10] = rv # rho_V

    return xformed
