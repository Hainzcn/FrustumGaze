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
    [Tooltip("视线数据缩放 (例如从厘米转米: 0.01)")]
    public float inputScale = 0.01f;
    [Tooltip("手部数据缩放 (通常为 1.0，因为源数据已经是米)")]
    public float handInputScale = 1.0f;

    [Header("Runtime Data (Read Only)")]
    [SerializeField] private Vector3 _gazeData; // Gaze: dist, x, y
    [SerializeField] private Vector3 _handData; // Hand: x, y, z
    [SerializeField] private bool _isPinching;

    // 公共访问属性 (线程安全读取)
    public Vector3 LatestData => _gazeData; // 兼容旧代码 (视线数据)
    public Vector3 GazeData => _gazeData;
    public Vector3 HandData => _handData;
    public bool IsPinching => _isPinching;

    // 内部网络变量
    private Socket _socket;
    private Thread _receiveThread;
    private bool _isRunning = false;
    private byte[] _receiveBuffer = new byte[1024]; // 复用缓冲区
    private EndPoint _remoteEndPoint;

    // 锁对象，用于同步 float 解析后的赋值 (比 lock 整个 socket 操作轻量得多)
    private readonly object _dataLock = new object();

    // 上采样插值相关变量
    private Vector3 _gazeNetworkBuffer; 
    private bool _hasNewGazeData = false;
    private Vector3 _gazeStart;
    private Vector3 _gazeEnd;
    private float _gazeTime = 0f;

    private Vector3 _handNetworkBuffer;
    private bool _handPinchBuffer;
    private bool _hasNewHandData = false;
    private Vector3 _handStart;
    private Vector3 _handEnd;
    private float _handTime = 0f;

    private const float TARGET_UPDATE_INTERVAL = 1.0f / 30.0f; // 假设源数据是 30Hz

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

    void Update()
    {
        // --- Gaze Interpolation ---
        bool hasNewGaze = false;
        Vector3 newGaze = Vector3.zero;

        // --- Hand Interpolation ---
        bool hasNewHand = false;
        Vector3 newHand = Vector3.zero;
        bool newPinch = false;

        lock (_dataLock)
        {
            if (_hasNewGazeData)
            {
                newGaze = _gazeNetworkBuffer;
                hasNewGaze = true;
                _hasNewGazeData = false;
            }
            if (_hasNewHandData)
            {
                newHand = _handNetworkBuffer;
                newPinch = _handPinchBuffer;
                hasNewHand = true;
                _hasNewHandData = false;
            }
        }

        // Gaze Update
        if (hasNewGaze)
        {
            _gazeStart = _gazeData; 
            _gazeEnd = newGaze;
            _gazeTime = 0f;
        }
        _gazeTime += Time.deltaTime;
        float tGaze = Mathf.Clamp01(_gazeTime / TARGET_UPDATE_INTERVAL);
        _gazeData = Vector3.Lerp(_gazeStart, _gazeEnd, tGaze);

        // Hand Update
        if (hasNewHand)
        {
            _handStart = _handData;
            _handEnd = newHand;
            _isPinching = newPinch; // 状态直接更新，不插值
            _handTime = 0f;
        }
        _handTime += Time.deltaTime;
        float tHand = Mathf.Clamp01(_handTime / TARGET_UPDATE_INTERVAL);
        _handData = Vector3.Lerp(_handStart, _handEnd, tHand);
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
            catch { }

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
            if (string.IsNullOrEmpty(text)) return;

            // 1. Hand Data (H:isPinch,x,y,z)
            if (text.StartsWith("H:"))
            {
                string content = text.Substring(2);
                string[] parts = content.Split(',');
                if (parts.Length >= 4)
                {
                    int isPinch = int.Parse(parts[0]);
                    float x = -float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture) * handInputScale;
                    float y = -float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture) * handInputScale;
                    float z = -float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture) * handInputScale;

                    lock (_dataLock)
                    {
                        _handNetworkBuffer = new Vector3(x, y, z);
                        _handPinchBuffer = (isPinch != 0);
                        _hasNewHandData = true;
                    }
                }
                return;
            }

            // 2. Gaze Data (G:z,x,y OR old format z,x,y)
            string gazeContent = text;
            if (text.StartsWith("G:"))
            {
                gazeContent = text.Substring(2);
            }

            // 解析 Gaze
            int firstComma = gazeContent.IndexOf(',');
            if (firstComma == -1) return;

            int secondComma = gazeContent.IndexOf(',', firstComma + 1);
            if (secondComma == -1) return;

            // 使用 Substring 解析 (或者用 Span<char> 如果是 .NET Standard 2.1)
            string sZ = gazeContent.Substring(0, firstComma);
            string sX = gazeContent.Substring(firstComma + 1, secondComma - firstComma - 1);
            string sY = gazeContent.Substring(secondComma + 1);

            // 使用 InvariantCulture 防止不同地区系统(如使用逗号小数点的地区)解析错误
            float gx = -float.Parse(sX, System.Globalization.CultureInfo.InvariantCulture) * inputScale;
            float gy = -float.Parse(sY, System.Globalization.CultureInfo.InvariantCulture) * inputScale;
            float gz = -float.Parse(sZ, System.Globalization.CultureInfo.InvariantCulture) * inputScale;

            // 写入数据
            // 由于 Vector3 赋值不是原子的，这里使用 lock 确保数据一致性
            lock (_dataLock)
            {
                _gazeNetworkBuffer = new Vector3(gx, gy, gz);
                _hasNewGazeData = true;
            }
        }
        catch
        {
            // 解析失败忽略
        }
    }
}
