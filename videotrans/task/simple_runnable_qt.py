try:
    from PySide6.QtCore import QRunnable, QThreadPool
except Exception:
    QRunnable = None
    QThreadPool = None

import threading
from videotrans import logger


if QRunnable is not None:
    class SimpleRunnable(QRunnable):
        def __init__(self, func, *args, **kwargs):
            super().__init__()
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def run(self):
            try:
                self.func(*self.args, **self.kwargs)
            except Exception as e:
                logger.exception(
                    f'后台线程执行任务失败:{self.args=},{self.kwargs=},{e}',
                    exc_info=True,
                )
else:
    class SimpleRunnable:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def run(self):
            try:
                self.func(*self.args, **self.kwargs)
            except Exception as e:
                logger.exception(
                    f'后台线程执行任务失败:{self.args=},{self.kwargs=},{e}',
                    exc_info=True,
                )


def run_in_threadpool(func, *args, **kwargs):
    if QThreadPool is not None:
        runnable = SimpleRunnable(func, *args, **kwargs)
        QThreadPool.globalInstance().start(runnable)
        return runnable

    thread = threading.Thread(
        target=SimpleRunnable(func, *args, **kwargs).run,
        daemon=True,
    )
    thread.start()
    return thread
