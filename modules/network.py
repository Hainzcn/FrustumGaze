import socket
import threading
import queue
from config import settings

class UDPSender:
    """
    Asynchronous UDP Sender.
    Uses a background thread and a queue to prevent network jitter from blocking the main thread.
    """
    def __init__(self, ip, port, queue_size=10):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = True
        
        # Start the sender thread
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()
        
        print(f"UDP socket initialized (Async). Target: {self.ip}:{self.port}")

    def _sender_loop(self):
        """Background thread loop for sending UDP packets."""
        while self.running:
            try:
                # Block until an item is available
                data_str = self.queue.get(timeout=0.1)
                if data_str is None: # Sentinel value for stopping
                    break
                    
                # Send the data
                self.sock.sendto(data_str.encode('utf-8'), (self.ip, self.port))
                
                if settings.PRINT_UDP_DATA:
                    print(f"UDP Sent: {data_str}")
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"UDP Send Error (in thread): {e}")

    def send(self, data_str):
        """
        Non-blocking send. Puts data into the queue.
        If the queue is full, the oldest item is discarded to make room for the new one (LIFO behavior for real-time data).
        """
        if not self.running:
            return

        try:
            self.queue.put_nowait(data_str)
        except queue.Full:
            # If queue is full, remove the oldest item and try again (drop strategy)
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(data_str)
            except queue.Empty:
                pass # Should not happen if full
            except queue.Full:
                pass # Still full, just drop the current packet

    def close(self):
        """Stops the sender thread and closes the socket."""
        self.running = False
        # Wake up the thread if it's waiting on get()
        try:
            self.queue.put(None) 
        except:
            pass
            
        if self.sender_thread.is_alive():
            self.sender_thread.join(timeout=1.0)
            
        self.sock.close()
