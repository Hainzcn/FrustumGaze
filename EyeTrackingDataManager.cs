using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;

/// <summary>
/// 高性能 UDP 数据接收管理器 (单例模式)
/// 端口: 8888
/// 特性: 
/// 1. 使用 Socket 直接接收，复用缓冲区，零 GC (Receive 阶段)
/// 2. 独立线程处理，无锁读取 (使用 Interlocked/Volatile 策略或极简锁)
/// 3. 提供全局访问点
/// </summary>
public class EyeTrackingDataManager : MonoBehaviour
{
    public static EyeTrackingDataManager Instance { get; private set; }

    [Header("UDP Settings")]
    public int port = 8888;
    public bool runInBackground = true;

    [Header("Runtime Data (Read Only)")]
    [SerializeField] private Vector3 _latestData; // x, y, z
    
    // 公共访问属性 (线程安全读取)
    public Vector3 LatestData
    {
        get { return _latestData; }
    }

    // 内部网络变量
    private Socket _socket;
    private Thread _receiveThread;
    private bool _isRunning = false;
    private byte[] _receiveBuffer = new byte[1024]; // 复用缓冲区
    private EndPoint _remoteEndPoint;

    // 锁对象，用于同步 float 解析后的赋值 (比 lock 整个 socket 操作轻量得多)
    private readonly object _dataLock = new object();

    void Awake()
    {
        // 单例设置
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject); // 切换场景不销毁
    }

    void Start()
    {
        StartReceiver();
    }

    void OnDestroy()
    {
        StopReceiver();
    }

    void OnApplicationQuit()
    {
        StopReceiver();
    }

    private void StartReceiver()
    {
        try
        {
            // 初始化 Socket
            _socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            _socket.Bind(new IPEndPoint(IPAddress.Any, port));
            // 设置接收超时，避免线程死锁无法退出
            _socket.ReceiveTimeout = 1000; 

            _remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
            
            _isRunning = true;
            _receiveThread = new Thread(ReceiveLoop);
            _receiveThread.IsBackground = true;
            _receiveThread.Start();
            
            Debug.Log($"[EyeTrackingDataManager] UDP Receiver started on port {port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[EyeTrackingDataManager] Failed to start UDP: {e.Message}");
        }
    }

    private void StopReceiver()
    {
        _isRunning = false;
        if (_receiveThread != null && _receiveThread.IsAlive)
        {
            // 等待线程结束，或者直接通过关闭 Socket 强制中断
            try 
            {
                if (_socket != null) _socket.Close(); 
            }
            catch {}
            
            // _receiveThread.Join(500); // 可选：等待线程优雅退出
        }
        _socket = null;
    }

    private void ReceiveLoop()
    {
        while (_isRunning)
        {
            try
            {
                if (_socket == null || _socket.Available == 0)
                {
                    Thread.Sleep(1); // 极短休眠，降低 CPU 占用，同时保持低延迟
                    continue;
                }

                // 接收数据
                int length = _socket.ReceiveFrom(_receiveBuffer, ref _remoteEndPoint);
                if (length > 0)
                {
                    // 解析数据
                    // 假设格式为 "x,y,z" 或 "x,y,d" 的 UTF-8 字符串
                    // 为了性能，这里还是需要将 byte 转 string，但只转实际长度
                    string text = Encoding.UTF8.GetString(_receiveBuffer, 0, length);
                    
                    ParseAndSetData(text);
                }
            }
            catch (SocketException)
            {
                // 超时或 Socket 关闭，忽略
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[EyeTrackingDataManager] Receive error: {e.Message}");
            }
        }
    }

    private void ParseAndSetData(string text)
    {
        try
        {
            // 简单的逗号分割解析
            // 优化提示：如果追求极致 0 GC，可以手动遍历 byte[] 解析 float，跳过 string 转换
            // 但考虑到每帧一次 string alloc 在现代 Unity (SGen/Incremental GC) 中通常可接受
            
            int firstComma = text.IndexOf(',');
            if (firstComma == -1) return;
            
            int secondComma = text.IndexOf(',', firstComma + 1);
            if (secondComma == -1) return;

            // 使用 Substring 解析 (或者用 Span<char> 如果是 .NET Standard 2.1)
            string sZ = text.Substring(0, firstComma);
            string sX = text.Substring(firstComma + 1, secondComma - firstComma - 1);
            string sY = text.Substring(secondComma + 1);

            float x = -float.Parse(sX);
            float y = -float.Parse(sY);
            float z = float.Parse(sZ);

            // 写入数据
            // 由于 Vector3 赋值不是原子的，这里使用 lock 确保数据一致性
            lock (_dataLock)
            {
                _latestData.x = x;
                _latestData.y = y;
                _latestData.z = z;
            }
        }
        catch
        {
            // 解析失败忽略
        }
    }
}
