# SPDX-License-Identifier: Apache-2.0
class NavatalaError(RuntimeError):
    pass

class InvalidHandleError(NavatalaError):
    pass

class GpuRuntimeError(NavatalaError):
    pass

class OutOfMemoryError(MemoryError, NavatalaError):
    pass

class TimeoutGpuError(TimeoutError, NavatalaError):
    pass

class CancelledError(NavatalaError):
    pass

class AlreadyExistsError(FileExistsError, NavatalaError):
    pass

class NotFoundError(FileNotFoundError, NavatalaError):
    pass

class PermissionDeniedError(PermissionError, NavatalaError):
    pass

class NavatalaIOError(OSError, NavatalaError):
    pass

class NavatalaInternalError(NavatalaError):
    pass

class InvalidShapeError(ValueError, NavatalaError):
    pass

class InvalidDTypeError(TypeError, NavatalaError):
    pass

class UnsupportedBackendError(NotImplementedError, NavatalaError):
    pass

class NonContiguousError(ValueError, NavatalaError):
    pass

class InvalidArgumentError(ValueError, NavatalaError):
    pass

class RuntimeFailureError(NavatalaError):
    pass

class MissingExtensionError(ImportError, NavatalaError):
    pass
