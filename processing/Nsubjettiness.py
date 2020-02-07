# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_Nsubjettiness')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_Nsubjettiness')
    _Nsubjettiness = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_Nsubjettiness', [dirname(__file__)])
        except ImportError:
            import _Nsubjettiness
            return _Nsubjettiness
        try:
            _mod = imp.load_module('_Nsubjettiness', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _Nsubjettiness = swig_import_helper()
    del swig_import_helper
else:
    import _Nsubjettiness
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class Nsubjettinesswrapper(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Nsubjettinesswrapper, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Nsubjettinesswrapper, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _Nsubjettiness.new_Nsubjettinesswrapper(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def getTau(self, maxTau, particles):
        return _Nsubjettiness.Nsubjettinesswrapper_getTau(self, maxTau, particles)
    NormalizedMeasure = _Nsubjettiness.Nsubjettinesswrapper_NormalizedMeasure
    UnnormalizedMeasure = _Nsubjettiness.Nsubjettinesswrapper_UnnormalizedMeasure
    OriginalGeometricMeasure = _Nsubjettiness.Nsubjettinesswrapper_OriginalGeometricMeasure
    NormalizedCutoffMeasure = _Nsubjettiness.Nsubjettinesswrapper_NormalizedCutoffMeasure
    UnnormalizedCutoffMeasure = _Nsubjettiness.Nsubjettinesswrapper_UnnormalizedCutoffMeasure
    GeometricCutoffMeasure = _Nsubjettiness.Nsubjettinesswrapper_GeometricCutoffMeasure
    N_MEASURE_DEFINITIONS = _Nsubjettiness.Nsubjettinesswrapper_N_MEASURE_DEFINITIONS
    KT_Axes = _Nsubjettiness.Nsubjettinesswrapper_KT_Axes
    CA_Axes = _Nsubjettiness.Nsubjettinesswrapper_CA_Axes
    AntiKT_Axes = _Nsubjettiness.Nsubjettinesswrapper_AntiKT_Axes
    WTA_KT_Axes = _Nsubjettiness.Nsubjettinesswrapper_WTA_KT_Axes
    WTA_CA_Axes = _Nsubjettiness.Nsubjettinesswrapper_WTA_CA_Axes
    Manual_Axes = _Nsubjettiness.Nsubjettinesswrapper_Manual_Axes
    OnePass_KT_Axes = _Nsubjettiness.Nsubjettinesswrapper_OnePass_KT_Axes
    OnePass_CA_Axes = _Nsubjettiness.Nsubjettinesswrapper_OnePass_CA_Axes
    OnePass_AntiKT_Axes = _Nsubjettiness.Nsubjettinesswrapper_OnePass_AntiKT_Axes
    OnePass_WTA_KT_Axes = _Nsubjettiness.Nsubjettinesswrapper_OnePass_WTA_KT_Axes
    OnePass_WTA_CA_Axes = _Nsubjettiness.Nsubjettinesswrapper_OnePass_WTA_CA_Axes
    OnePass_Manual_Axes = _Nsubjettiness.Nsubjettinesswrapper_OnePass_Manual_Axes
    MultiPass_Axes = _Nsubjettiness.Nsubjettinesswrapper_MultiPass_Axes
    N_AXES_DEFINITIONS = _Nsubjettiness.Nsubjettinesswrapper_N_AXES_DEFINITIONS
    __swig_destroy__ = _Nsubjettiness.delete_Nsubjettinesswrapper
    __del__ = lambda self: None
Nsubjettinesswrapper_swigregister = _Nsubjettiness.Nsubjettinesswrapper_swigregister
Nsubjettinesswrapper_swigregister(Nsubjettinesswrapper)

# This file is compatible with both classic and new-style classes.

