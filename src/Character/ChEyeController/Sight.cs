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
    // [Tooltip("输入数据的单位缩放")]
    // public float dataScale = 1.0f; // 已移至 DataManager 统一管理

    [Header("限制参数")]
    [Tooltip("最大旋转角度")]
    [Range(0, 180)] public float maxAngle = 60f;
    [Tooltip("旋转平滑速度")]
    public float smoothing = 20f;

    [Header("生动性设置 (Liveness)")]
    [Tooltip("启用眼球微颤动 (Micro-saccades)")]
    public bool enableMicroJitter = true;
    [Tooltip("微颤动幅度 (角度)")]
    [Range(0f, 2f)] public float jitterStrength = 0.15f;
    [Tooltip("微颤动更新频率 (Hz)")]
    [Range(1f, 60f)] public float jitterFrequency = 20f;

    [Tooltip("启用随机瞥视 (Saccades)")]
    public bool enableSaccades = true;
    [Tooltip("平均每分钟瞥视次数")]
    public float saccadeFrequency = 10f;
    [Tooltip("瞥视持续时间 (秒)")]
    public Vector2 saccadeDuration = new Vector2(0.1f, 0.3f);
    [Tooltip("瞥视幅度 (角度)")]
    public float saccadeAngle = 10f;
    [Tooltip("瞥视时的速度倍率")]
    public float saccadeSpeedMultiplier = 3.0f;

    // 内部计算用的偏移量 (用于自动匹配骨骼的初始轴向)
    private Quaternion leftEyeOffset;
    private Quaternion rightEyeOffset;
    private bool isInitialized = false;

    // 生动性内部变量
    private Vector3 currentJitterOffset;
    private Vector3 currentSaccadeOffset;
    private float nextSaccadeTime;
    private float saccadeEndTime;
    private bool isSaccading;
    private float lastJitterTime;
    private float currentSpeedMultiplier = 1.0f;

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

        // 初始化瞥视时间
        ScheduleNextSaccade();

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
            // 注意：DataManager 中已经应用了缩放 (inputScale)，这里直接使用 magnitude 即可
            float d = rawData.magnitude;
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

        // 更新生动性状态
        if (Application.isPlaying)
        {
            UpdateLiveness();
        }

        // 计算包含生动性偏移的最终目标点
        Vector3 finalTargetPos = GetLivenessTarget(target.position);

        if (leftEye != null) RotateEye(leftEye, leftEyeOffset, finalTargetPos);
        if (rightEye != null) RotateEye(rightEye, rightEyeOffset, finalTargetPos);
    }

    void UpdateLiveness()
    {
        currentSpeedMultiplier = 1.0f;

        // 1. 微颤动 (Jitter)
        if (enableMicroJitter)
        {
            if (Time.time - lastJitterTime > (1f / jitterFrequency))
            {
                lastJitterTime = Time.time;
                // 在单位圆内随机取点，模拟微小的角度偏移
                currentJitterOffset = UnityEngine.Random.insideUnitCircle * jitterStrength;
            }
        }
        else
        {
            currentJitterOffset = Vector3.zero;
        }

        // 2. 瞥视 (Saccades)
        if (enableSaccades)
        {
            if (!isSaccading)
            {
                // 检查是否应该开始瞥视
                if (Time.time >= nextSaccadeTime)
                {
                    isSaccading = true;
                    float duration = UnityEngine.Random.Range(saccadeDuration.x, saccadeDuration.y);
                    saccadeEndTime = Time.time + duration;
                    
                    // 随机选择一个方向进行瞥视
                    Vector2 rnd = UnityEngine.Random.insideUnitCircle.normalized;
                    currentSaccadeOffset = rnd * saccadeAngle;
                    
                    // 瞥视时眼球移动非常快
                    currentSpeedMultiplier = saccadeSpeedMultiplier;
                }
            }
            else
            {
                // 正在瞥视中
                if (Time.time >= saccadeEndTime)
                {
                    // 结束瞥视，回到主目标
                    isSaccading = false;
                    ScheduleNextSaccade();
                    currentSaccadeOffset = Vector3.zero;
                }
                else
                {
                    // 保持快速移动状态
                    currentSpeedMultiplier = saccadeSpeedMultiplier;
                }
            }
        }
    }

    void ScheduleNextSaccade()
    {
        if (saccadeFrequency <= 0) 
        {
            nextSaccadeTime = float.MaxValue;
            return;
        }
        // 计算下一次瞥视的延迟
        float avgDelay = 60f / saccadeFrequency;
        // 添加随机性 (+/- 40%)
        float delay = avgDelay * UnityEngine.Random.Range(0.6f, 1.4f);
        nextSaccadeTime = Time.time + delay;
    }

    Vector3 GetLivenessTarget(Vector3 baseTargetPos)
    {
        // 头部位置
        Vector3 headPos = transform.position;
        // 原始注视方向
        Vector3 baseDir = baseTargetPos - headPos;
        float distance = baseDir.magnitude;
        if (distance < 0.01f) distance = 1f; // 防止过近

        // 计算基础旋转 (Looking at target)
        // 使用 transform.up 作为参考上方向
        Quaternion baseLook = Quaternion.LookRotation(baseDir, transform.up);
        
        // 叠加偏移 (Euler Angles)
        // Jitter和SaccadeOffset也是Vector3/Vector2，这里视作 X/Y 轴的角度偏移
        float xAngleOffset = currentJitterOffset.x + currentSaccadeOffset.x;
        float yAngleOffset = currentJitterOffset.y + currentSaccadeOffset.y;
        
        // 注意：在局部坐标系下，绕Y轴旋转是左右(Yaw)，绕X轴旋转是上下(Pitch)
        Quaternion deflection = Quaternion.Euler(-yAngleOffset, xAngleOffset, 0);
        
        // 最终方向 = 基础旋转 * 偏转
        Vector3 finalDir = (baseLook * deflection) * Vector3.forward;
        
        // 还原为世界坐标点 (保持距离不变)
        return headPos + finalDir * distance;
    }

    void RotateEye(Transform eye, Quaternion offset, Vector3 targetPos)
    {
        Vector3 dir = targetPos - eye.position;
        if (dir == Vector3.zero) return;

        // A. 角度限制 (Clamp)
        Vector3 headForward = transform.forward;
        float angle = Vector3.Angle(headForward, dir);
        if (angle > maxAngle)
        {
            dir = Vector3.RotateTowards(headForward, dir, maxAngle * Mathf.Deg2Rad, 0.0f);
        }

        // B. 计算标准的注视旋转
        Quaternion standardLook = Quaternion.LookRotation(dir, transform.up);

        // C. 应用初始偏移
        Quaternion targetRot = standardLook * offset;

        // D. 平滑插值 (应用速度倍率)
        float currentSmoothing = smoothing * currentSpeedMultiplier;
        eye.rotation = Quaternion.Slerp(eye.rotation, targetRot, Time.deltaTime * currentSmoothing);
    }
}
