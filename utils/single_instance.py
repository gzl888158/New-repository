"""
跨平台单实例锁，防止同一时间运行多个程序实例。
"""
import os
import sys
import socket
import errno
from loguru import logger


class SingleInstance:
    """跨平台单实例锁，防止多进程同时运行。

    Windows: 使用命名互斥量 (CreateMutex)，进程退出后自动释放，最可靠。
    Linux/Mac: 使用本地 TCP 端口绑定作为互斥锁。
    同时保留文件锁作为辅助诊断手段。
    """

    def __init__(self, mutex_name: str = "Global\\OKX_Quant_Trading_SingleInstance",
                 port: int = 29333, lock_file: str = "data/app.lock"):
        self._mutex_name = mutex_name
        self._port = port
        self.lock_file = lock_file
        self._socket: socket.socket = None
        self._fh = None
        self._mutex_handle = None

    def acquire(self) -> bool:
        """尝试获取锁，成功返回True，已有实例运行返回False"""
        dir_path = os.path.dirname(self.lock_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.windll.kernel32
                CreateMutex = kernel32.CreateMutexW
                CreateMutex.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
                CreateMutex.restype = wintypes.HANDLE
                GetLastError = kernel32.GetLastError
                ERROR_ALREADY_EXISTS = 183

                self._mutex_handle = CreateMutex(None, False, self._mutex_name)
                if not self._mutex_handle:
                    logger.error("Failed to create single-instance mutex")
                    return False
                if GetLastError() == ERROR_ALREADY_EXISTS:
                    logger.error("Another instance is already running (mutex already exists)")
                    kernel32.CloseHandle(self._mutex_handle)
                    self._mutex_handle = None
                    return False
                # 写入PID到锁文件，便于人工排查
                self._write_lock_file()
                return True
            except Exception as e:
                logger.error(f"Single-instance mutex error: {e}")
                return False
        else:
            # Linux/Mac: 使用本地端口绑定
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 不设置 SO_REUSEADDR，确保同一端口只能绑定一次
                self._socket.bind(("127.0.0.1", self._port))
                self._socket.listen(1)
            except socket.error as e:
                if e.errno in (errno.EADDRINUSE,):
                    logger.error(f"Another instance is already running (port {self._port} in use)")
                else:
                    logger.error(f"Failed to acquire single-instance lock: {e}")
                if self._socket:
                    self._socket.close()
                    self._socket = None
                return False
            self._write_lock_file()
            return True

    def _write_lock_file(self):
        """将PID写入锁文件，仅作辅助诊断"""
        try:
            with open(self.lock_file, "w") as f:
                f.write(str(os.getpid()))
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to write lock file: {e}")

    def release(self):
        """释放锁"""
        if sys.platform == "win32" and self._mutex_handle:
            try:
                import ctypes
                from ctypes import wintypes
                ctypes.windll.kernel32.CloseHandle(self._mutex_handle)
            except Exception as e:
                logger.warning(f"Error releasing mutex: {e}")
            finally:
                self._mutex_handle = None

        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.warning(f"Error releasing socket lock: {e}")
            finally:
                self._socket = None

        if self._fh:
            try:
                self._fh.close()
                try:
                    os.remove(self.lock_file)
                except OSError:
                    pass
            except Exception as e:
                logger.warning(f"Error releasing file lock: {e}")
            finally:
                self._fh = None
