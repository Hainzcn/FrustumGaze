
import multiprocessing
from multiprocessing import shared_memory
import numpy as np

class SharedMemoryManager:
    def __init__(self, name, size, create=True):
        self.name = name
        self.size = size
        self.shm = None
        self.create = create
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=self.size, name=self.name)
            except FileExistsError:
                # 如果已存在，尝试连接并重建（或者复用）
                # 这里简单策略：先unlink再create
                try:
                    existing = shared_memory.SharedMemory(name=self.name)
                    existing.close()
                    existing.unlink()
                except:
                    pass
                self.shm = shared_memory.SharedMemory(create=True, size=self.size, name=self.name)
        else:
            self.shm = shared_memory.SharedMemory(name=self.name)

    def get_buffer(self):
        return self.shm.buf

    def close(self):
        if self.shm:
            self.shm.close()

    def unlink(self):
        if self.shm and self.create:
            try:
                self.shm.unlink()
            except:
                pass

def create_shared_array(shape, dtype=np.uint8, name="frame_shm"):
    """
    辅助函数：创建共享内存并返回对应的 numpy 数组视图
    """
    # 计算字节大小
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    
    manager = SharedMemoryManager(name, size, create=True)
    array = np.ndarray(shape, dtype=dtype, buffer=manager.get_buffer())
    
    return manager, array

def get_shared_array(name, shape, dtype=np.uint8):
    """
    辅助函数：获取已存在的共享内存的 numpy 数组视图
    """
    # 计算字节大小
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    
    manager = SharedMemoryManager(name, size, create=False)
    array = np.ndarray(shape, dtype=dtype, buffer=manager.get_buffer())
    
    return manager, array
