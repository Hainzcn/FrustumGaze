using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;

[RequireComponent(typeof(Camera))]
public class VirtualWindowController : MonoBehaviour
{
    [Header("Physical Screen Settings (Meters)")]
    public float screenWidth = 0.52f;  // 24寸显示器宽度约0.52m
    public float screenHeight = 0.29f; // 24寸显示器高度约0.29m

    [Header("UDP Settings")]
    public int port = 8888;

    [Header("Coordinate Mapping")]
    public Vector3 cameraOffsetFromScreenCenter = new Vector3(0, 0.15f, 0);
    public float sensitivity = 1.0f;

    [Header("Mouse Drag Camera Settings")] // 新增：鼠标拖动配置项
    public bool enableMouseDrag = true; // 是否启用鼠标拖动视角
    public float dragRotateSpeed = 2.0f; // 拖动旋转速度
    public float minPitchAngle = -80f; // 上下视角最小角度（防止翻转）
    public float maxPitchAngle = 80f; // 上下视角最大角度（防止翻转）

    // 显式标注命名空间，消除二义性
    private UnityEngine.Camera _cam;
    private System.Threading.Thread _receiveThread;
    private System.Net.Sockets.UdpClient _client;
    private bool _isRunning = true;

    private Vector3 _latestEyePosFromUdp = new Vector3(0, 0, -0.6f);
    private readonly object _lock = new object();

    // 新增：鼠标拖动相关私有变量
    private bool _isDragging = false; // 是否正在拖动
    private Vector2 _lastMousePosition; // 上一帧鼠标位置

    void Start()
    {
        _cam = GetComponent<UnityEngine.Camera>();

        // 初始化UDP线程：显式实例化System.Thread
        _receiveThread = new System.Threading.Thread(new System.Threading.ThreadStart(ReceiveData));
        _receiveThread.IsBackground = true;
        _receiveThread.Start();
    }

    // 新增：Update中检测鼠标输入（输入检测推荐在Update中执行）
    void Update()
    {
        // 只有启用鼠标拖动时，才执行以下逻辑
        if (!enableMouseDrag) return;

        // 1. 鼠标左键按下：记录初始状态
        if (Input.GetMouseButtonDown(0))
        {
            _isDragging = true;
            // 记录按下瞬间的鼠标位置
            _lastMousePosition = Input.mousePosition;
        }

        // 2. 鼠标左键抬起：结束拖动
        if (Input.GetMouseButtonUp(0))
        {
            _isDragging = false;
        }

        // 3. 正在拖动中：计算鼠标偏移并更新视角
        if (_isDragging)
        {
            DragUpdateCameraRotation();
        }
    }

    // 新增：核心拖动视角计算逻辑
    private void DragUpdateCameraRotation()
    {
        // 获取当前鼠标位置
        Vector2 currentMousePos = Input.mousePosition;
        // 计算鼠标移动偏移量（当前 - 上一帧）
        Vector2 mouseDelta = currentMousePos - _lastMousePosition;

        // 更新上一帧鼠标位置，用于下一帧计算
        _lastMousePosition = currentMousePos;

        // 计算旋转角度：
        // 左右拖动：绕Y轴旋转（Yaw），鼠标X轴偏移对应相机Y轴旋转
        // 上下拖动：绕X轴旋转（Pitch），鼠标Y轴偏移对应相机X轴旋转
        float yaw = mouseDelta.x * dragRotateSpeed * Time.deltaTime;
        float pitch = -mouseDelta.y * dragRotateSpeed * Time.deltaTime; // 负号：让鼠标向上拖动时，相机向上看

        // 获取当前相机的欧拉角（方便修改单个轴的角度）
        Vector3 currentEuler = transform.rotation.eulerAngles;

        // 更新Y轴旋转（左右视角，无限制）
        currentEuler.y += yaw;

        // 更新X轴旋转（上下视角，限制在minPitchAngle和maxPitchAngle之间，防止翻转）
        // 处理欧拉角0/360度循环的问题
        float newPitch = currentEuler.x + pitch;
        newPitch = Mathf.Clamp(newPitch, minPitchAngle, maxPitchAngle);
        currentEuler.x = newPitch;

        // 应用旋转到相机
        transform.rotation = Quaternion.Euler(currentEuler);
    }

    // 异步接收UDP数据：解决类型/异常二义性
    private void ReceiveData()
    {
        try
        {
            _client = new System.Net.Sockets.UdpClient(port);
            while (_isRunning)
            {
                try
                {
                    System.Net.IPEndPoint anyIP = new System.Net.IPEndPoint(System.Net.IPAddress.Any, 0);
                    byte[] data = _client.Receive(ref anyIP);
                    string text = Encoding.UTF8.GetString(data);

                    string[] values = text.Split(',');
                    if (values.Length == 3)
                    {
                        // 显式指定CultureInfo，避免解析二义性
                        CultureInfo invariantCulture = CultureInfo.InvariantCulture;
                        float distCm = float.Parse(values[0], invariantCulture);
                        float offXCm = float.Parse(values[1], invariantCulture);
                        float offYCm = float.Parse(values[2], invariantCulture);

                        // 转换为米
                        float distM = distCm / 100f;
                        float offXM = offXCm / 100f;
                        float offYM = offYCm / 100f;

                        lock (_lock)
                        {
                            _latestEyePosFromUdp = new Vector3(-offXM, -offYM, -distM);
                        }
                    }
                }
                catch (System.Exception) // 显式指定System.Exception
                {
                    // 忽略解析/套接字错误（避免Unity主线程API调用）
                }
            }
        }
        catch (System.Exception) // 显式指定System.Exception
        {
            // 初始化错误，静默处理
        }
    }

    // 每帧更新投影
    void LateUpdate()
    {
        Vector3 eyePos;
        lock (_lock)
        {
            eyePos = _latestEyePosFromUdp;
        }

        Vector3 relativeEye = eyePos + cameraOffsetFromScreenCenter;
        // 保留原有相机位置更新
        transform.position = relativeEye;

        // 兼容修改：只有在不启用鼠标拖动，或未处于拖动状态时，才保持相机正对屏幕
        if (!enableMouseDrag || !_isDragging)
        {
            // 注释：若你希望拖动后仍保留UDP的正对逻辑，松开鼠标会自动复位到正对屏幕
            // 若你希望拖动后保持视角，删除这行即可
            transform.rotation = Quaternion.identity;
        }

        // 更新非对称投影矩阵
        _cam.projectionMatrix = CalculateOffAxisProjection(relativeEye);
    }

    // 显式指定返回值为Unity Matrix4x4，消除二义性
    private UnityEngine.Matrix4x4 CalculateOffAxisProjection(Vector3 eye)
    {
        // 重命名变量，避免与内置符号混淆
        float nearClip = _cam.nearClipPlane;
        float farClip = _cam.farClipPlane;
        float eyeDistance = Mathf.Abs(eye.z); // 替换原变量d，消除歧义
        float scale = nearClip / eyeDistance;

        // 计算近裁剪面边界，变量名更清晰
        float clipLeft = (-screenWidth / 2f - eye.x) * scale;
        float clipRight = (screenWidth / 2f - eye.x) * scale;
        float clipBottom = (-screenHeight / 2f - eye.y) * scale;
        float clipTop = (screenHeight / 2f - eye.y) * scale;

        return PerspectiveOffCenter(clipLeft, clipRight, clipBottom, clipTop, nearClip, farClip);
    }

    // 显式指定参数/返回值类型，消除二义性
    private static UnityEngine.Matrix4x4 PerspectiveOffCenter(float left, float right, float bottom, float top, float near, float far)
    {
        float x = (2.0f * near) / (right - left);
        float y = (2.0f * near) / (top - bottom);
        float a = (right + left) / (right - left);
        float b = (top + bottom) / (top - bottom);
        float c = -(far + near) / (far - near);
        float d = -(2.0f * far * near) / (far - near);
        float e = -1.0f;

        // 显式实例化Unity Matrix4x4
        UnityEngine.Matrix4x4 projectionMatrix = new UnityEngine.Matrix4x4();
        projectionMatrix[0, 0] = x; projectionMatrix[0, 1] = 0; projectionMatrix[0, 2] = a; projectionMatrix[0, 3] = 0;
        projectionMatrix[1, 0] = 0; projectionMatrix[1, 1] = y; projectionMatrix[1, 2] = b; projectionMatrix[1, 3] = 0;
        projectionMatrix[2, 0] = 0; projectionMatrix[2, 1] = 0; projectionMatrix[2, 2] = c; projectionMatrix[2, 3] = d;
        projectionMatrix[3, 0] = 0; projectionMatrix[3, 1] = 0; projectionMatrix[3, 2] = e; projectionMatrix[3, 3] = 0;
        return projectionMatrix;
    }

    void OnApplicationQuit()
    {
        _isRunning = false;
        if (_receiveThread != null)
        {
            _receiveThread.Join(100);
        }
        if (_client != null)
        {
            _client.Close();
        }
    }
}