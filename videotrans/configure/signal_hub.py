from typing import Union

from videotrans.task.taskcfg import SignMsg


try:
    from PySide6.QtCore import QObject, Signal, Slot
except Exception:
    QObject = None
    Signal = None

    def Slot(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator


class _HeadlessSignal:
    """Tiny Qt Signal-compatible fallback for Gradio/CLI/headless runtimes."""

    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)
        return callback

    def disconnect(self, callback=None):
        if callback is None:
            self._callbacks.clear()
            return
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


if QObject is not None:
    class SignalHub(QObject):
        """Singleton Qt signal center used by the desktop application."""

        _instance = None
        new_message = Signal(str, object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._initialized = True

        @classmethod
        def instance(cls) -> "SignalHub":
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        @Slot(str, object)
        def post(self, uuid: Union[str, None] = None, data: SignMsg = None):
            self.new_message.emit(uuid, data)

else:
    class SignalHub:
        """Headless SignalHub for Gradio, Kaggle and CLI environments.

        It intentionally mirrors the subset of Qt Signal behavior used by
        pyVideoTrans: .connect(), .disconnect(), and .emit().
        """

        _instance = None

        def __init__(self, parent=None):
            self._initialized = True
            self.new_message = _HeadlessSignal()

        @classmethod
        def instance(cls) -> "SignalHub":
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def post(self, uuid: Union[str, None] = None, data: SignMsg = None):
            self.new_message.emit(uuid, data)
