using UnityEngine;
using System;
using System.Text;
using System.Globalization;

[RequireComponent(typeof(Camera))]
public class VirtualWindowController : MonoBehaviour
{
    [Header("Physical Screen Settings (Meters)")]
    public float screenWidth = 0.35f;  // 24寸显示器宽度约0.52m
    public float screenHeight = 0.22f; // 24寸显示器高度约0.29m

    [Header("Coordinate Mapping")]
    public Vector3 cameraOffsetFromScreenCenter = new Vector3(0, 0.15f, 0);
    public float sensitivity = 1.0f;
    // public float unitScale = 0.01f; // 已移至 DataManager 统一管理

    [Header("Movement Smoothing")]
    [Tooltip("是否启用平滑移动")]
    public bool enableSmoothing = true;
    [Tooltip("平滑速度")]
    public float smoothingSpeed = 15.0f;

    // 显式标注命名空间，消除二义性
    private UnityEngine.Camera _cam;
    private Vector3 _virtualScreenCenter;
    
    void Start()
    {
        _cam = GetComponent<UnityEngine.Camera>();

        // 初始化虚拟屏幕中心
        // 假设在 Start 时，Camera 处于“默认观察位置”
        // 我们根据默认的 eyePos 反推屏幕中心在世界坐标系的位置
        // 这样可以保留用户在 Editor 中设置的 Camera 位置，防止运行时乱飞
        
        // 注意：DataManager.Instance 可能在 Start 时尚未准备好，或者我们想要一个硬编码的默认值
        // 这里使用默认的 60cm (0.6m) 作为初始距离假设。
        Vector3 defaultEyePos = new Vector3(0, 0, -0.6f); 
        Vector3 localEyePos = defaultEyePos + cameraOffsetFromScreenCenter;
        _virtualScreenCenter = transform.position - localEyePos;
        
        Debug.Log($"[VirtualWindow] Initialized. Screen Center inferred at: {_virtualScreenCenter}");
    }

    // 每帧更新投影
    void LateUpdate()
    {
        // 1. 获取眼球相对于屏幕的局部坐标 (Local Eye Position)
        // 默认值 (单位：米)
        Vector3 rawEyeData = new Vector3(0, 0, -0.6f);
        
        // 从 DataManager 获取数据
        if (EyeTrackingDataManager.Instance != null)
        {
            Vector3 data = EyeTrackingDataManager.Instance.LatestData;
            // 只有当接收到有效数据时(Z!=0)才覆盖默认值
            // 注意：data 已经被 DataManager 缩放过，单位应该是米
            if (Mathf.Abs(data.z) > 0.001f)
            {
                // 如果发现实际运行时 Z 轴反向，请尝试改为 -data.z
                rawEyeData = new Vector3(data.x, data.y, data.z);
            }
        }
        
        // 应用单位缩放 (已在 DataManager 处理，这里不再缩放)
        Vector3 eyePosRelative = rawEyeData;

        Vector3 targetPos = eyePosRelative + cameraOffsetFromScreenCenter;
        
        // 更新相机位置
        if (enableSmoothing)
        {
            transform.position = Vector3.Lerp(transform.position, targetPos, Time.deltaTime * smoothingSpeed);
        }
        else
        {
            transform.position = targetPos;
        }
        
        // 始终保持相机正对屏幕 (离轴投影要求相机平面平行于屏幕平面)
        transform.rotation = Quaternion.identity;

        // 更新非对称投影矩阵
        // 使用当前的 transform.position (已包含平滑效果)
        _cam.projectionMatrix = CalculateOffAxisProjection(transform.position);
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
}
