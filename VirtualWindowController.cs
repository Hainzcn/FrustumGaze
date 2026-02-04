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
    public float unitScale = 0.01f; // 厘米转米 (如果UDP数据单位是厘米)

    [Header("Movement Smoothing")]
    [Tooltip("是否启用平滑移动")]
    public bool enableSmoothing = true;
    [Tooltip("平滑速度")]
    public float smoothingSpeed = 15.0f;

    // 显式标注命名空间，消除二义性
    private UnityEngine.Camera _cam;
    
    void Start()
    {
        _cam = GetComponent<UnityEngine.Camera>();
    }

    // 每帧更新投影
    void LateUpdate()
    {
        Vector3 eyePos = new Vector3(0, 0, -0.6f); // 默认值

        // 改为从 DataManager 获取数据
        if (EyeTrackingDataManager.Instance != null)
        {
            Vector3 data = EyeTrackingDataManager.Instance.LatestData;
            // 假设 DataManager 返回的是 (x, y, d) 格式，通常单位为厘米
            // 映射为: (x, y, d) * unitScale
            // 依据用户反馈保持当前映射逻辑
            eyePos = new Vector3(data.x, data.y, data.z) * unitScale;
        }

        Vector3 targetPos = eyePos + cameraOffsetFromScreenCenter;
        
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
