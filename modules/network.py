
import socket

class UDPSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP socket initialized. Target: {self.ip}:{self.port}")

    def send(self, data_str):
        try:
            self.sock.sendto(data_str.encode('utf-8'), (self.ip, self.port))
        except Exception as e:
            print(f"UDP Send Error: {e}")

    def close(self):
        self.sock.close()
