using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;

public class EyeGazeFocus : MonoBehaviour
{
    [Header("眼球骨骼绑定")]
    public Transform leftEye;
    public Transform rightEye;

    [Header("目标设置")]
    [Tooltip("注视目标物体。如果启用UDP，此物体的坐标将被自动更新。")]
    public Transform target;

    [Header("数据源设置")]
    [Tooltip("是否启用眼动数据驱动")]
    public bool useEyeData = true;
    [Tooltip("输入数据的单位缩放")]
    public float dataScale = 1.0f;

    [Header("限制参数")]
    [Tooltip("最大旋转角度")]
    [Range(0, 180)] public float maxAngle = 60f;
    [Tooltip("旋转平滑速度")]
    public float smoothing = 20f;

    // 内部计算用的偏移量 (用于自动匹配骨骼的初始轴向)
    private Quaternion leftEyeOffset;
    private Quaternion rightEyeOffset;
    private bool isInitialized = false;

    void Start()
    {
        // 1. 目标初始化
        if (target == null)
        {
            GameObject t = new GameObject("EyeGazeTarget");
            target = t.transform;
            if (Camera.main != null)
            {
                // 默认看向主摄像机
                target.position = Camera.main.transform.position;
            }
            else
            {
                target.position = transform.position + transform.forward * 2f;
            }
        }

        // 2. 计算骨骼的初始偏移 (Rest Pose Offset)
        // 核心思路：假设在Start时刻，眼睛是处于“正视前方”的初始状态。
        // 我们计算从“标准正视旋转”到“当前骨骼旋转”的偏移量。
        // 这样无论骨骼的轴向是Y轴向前还是其他奇特角度（如 -99.43, 78.22, 78.22），都能被自动兼容，
        // 从而避免手动设置轴向导致的参考系漂移问题。
        
        // 参考基准：以父节点(头部)的前方和上方为基准的标准LookRotation
        Quaternion refRot = Quaternion.LookRotation(transform.forward, transform.up);

        if (leftEye != null)
            leftEyeOffset = Quaternion.Inverse(refRot) * leftEye.rotation;
        
        if (rightEye != null)
            rightEyeOffset = Quaternion.Inverse(refRot) * rightEye.rotation;

        isInitialized = true;
    }

    void Update()
    {
        // 在主线程更新 Target 位置 (保留原有的UDP逻辑)
        if (useEyeData && target != null && Camera.main != null && EyeTrackingDataManager.Instance != null)
        {
            Vector3 rawData = EyeTrackingDataManager.Instance.LatestData;
            
            // 计算距离 (假设z是深度)
            float d = rawData.magnitude * dataScale;
            if (d < 0.1f) d = 1.0f;

            // 保持原逻辑：人物看向Camera后方d距离的点
            Transform camTr = Camera.main.transform;
            Vector3 worldPos = camTr.position - camTr.forward * d;

            target.position = Vector3.Lerp(target.position, worldPos, Time.deltaTime * 10f); 
        }
    }

    void LateUpdate()
    {
        if (!isInitialized || target == null) return;

        if (leftEye != null) RotateEye(leftEye, leftEyeOffset);
        if (rightEye != null) RotateEye(rightEye, rightEyeOffset);
    }

    void RotateEye(Transform eye, Quaternion offset)
    {
        Vector3 dir = target.position - eye.position;
        if (dir == Vector3.zero) return;

        // A. 角度限制 (Clamp)
        // 在计算LookRotation之前，先限制方向向量与头部前方的夹角
        // 这样可以防止眼球翻转到脑后
        Vector3 headForward = transform.forward;
        float angle = Vector3.Angle(headForward, dir);
        if (angle > maxAngle)
        {
            // 将目标方向限制在最大角度的圆锥体内
            dir = Vector3.RotateTowards(headForward, dir, maxAngle * Mathf.Deg2Rad, 0.0f);
        }

        // B. 计算标准的注视旋转 (World Space)
        // 使用 transform.up (头部上方) 作为 Up Vector，防止眼球产生不自然的滚动(Reference Frame Drift)
        Quaternion standardLook = Quaternion.LookRotation(dir, transform.up);

        // C. 应用初始偏移
        // 最终旋转 = 标准注视旋转 * 初始偏移
        // 这会将“标准Z轴向前”映射回“骨骼定义的Y轴向前”以及其特有的旋转偏置
        Quaternion targetRot = standardLook * offset;

        // D. 平滑插值
        eye.rotation = Quaternion.Slerp(eye.rotation, targetRot, Time.deltaTime * smoothing);
    }
}
