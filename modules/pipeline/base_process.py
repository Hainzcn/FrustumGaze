import multiprocessing
import queue
import signal
import traceback
from modules.shared_mem import get_shared_array


class BaseProcessorProcess(multiprocessing.Process):
    """
    子进程基类，封装共享内存连接、主循环、队列交互、生命周期管理。
    子类只需实现:
      - on_init() -> bool : 初始化子进程本地资源 (tracker 等)，返回 True 表示成功
      - on_process(task, frame) -> dict | None : 处理单帧，返回结果或 None 跳过
      - on_cleanup() : 释放子进程本地资源 (可选)
    """

    PROCESS_NAME = "BaseProcessor"

    def __init__(self, input_queue, output_queue, stop_event,
                 shm_names, frame_shape, triple_buffer_idx=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.shm_names = shm_names
        self.frame_shape = frame_shape
        self.triple_buffer_idx = triple_buffer_idx
        self.daemon = True

        self.shm_managers = []
        self.shm_arrays = []

    # ---- 子类钩子 ----

    def on_init(self) -> bool:
        """子进程启动后调用，初始化 tracker 等重型资源。"""
        return True

    def on_process(self, task: dict, frame) -> dict | None:
        """处理单帧。返回结果 dict 放入输出队列，返回 None 则跳过。"""
        raise NotImplementedError

    def on_cleanup(self):
        """子进程退出前调用，释放 tracker 等资源。"""
        pass

    # ---- 生命周期 ----

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if not self._connect_shared_memory():
            return
        if not self.on_init():
            return

        print(f"{self.PROCESS_NAME}: 进程已启动并就绪。")
        self._main_loop()

        self.on_cleanup()
        self._close_shared_memory()

    def _connect_shared_memory(self) -> bool:
        names = self.shm_names if isinstance(self.shm_names, list) else [self.shm_names]
        for name in names:
            try:
                mgr, arr = get_shared_array(name, self.frame_shape)
                self.shm_managers.append(mgr)
                self.shm_arrays.append(arr)
            except Exception as e:
                print(f"{self.PROCESS_NAME}: Failed to connect to shared memory {name}: {e}")
                return False
        return True

    def _read_frame(self, task):
        if self.triple_buffer_idx is not None:
            read_idx = self.triple_buffer_idx.value
        else:
            read_idx = task.get('buffer_idx', 0)
        if 0 <= read_idx < len(self.shm_arrays):
            return self.shm_arrays[read_idx]
        return self.shm_arrays[0]

    def _send_result(self, result):
        if self.output_queue.full():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
        self.output_queue.put(result)

    def _main_loop(self):
        while not self.stop_event.is_set():
            try:
                try:
                    task = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame = self._read_frame(task)
                result = self.on_process(task, frame)

                if result is not None:
                    self._send_result(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"{self.PROCESS_NAME} Error: {e}")
                traceback.print_exc()

    def _close_shared_memory(self):
        for mgr in self.shm_managers:
            try:
                mgr.close()
            except Exception:
                pass
