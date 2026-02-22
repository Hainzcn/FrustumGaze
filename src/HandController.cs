using UnityEngine;

/// <summary>
/// 手部控制器
/// 根据 EyeTrackingDataManager 中的手部数据控制物体移动
/// </summary>
public class HandController : MonoBehaviour
{
    [Header("Configuration")]
    [Tooltip("物体初始位置偏移 (Object Initial Position Offset)")]
    public Vector3 initialPosition = Vector3.zero;

    [Tooltip("坐标缩放比例 (Coordinate Scale)")]
    public Vector3 scale = Vector3.one;

    [Tooltip("平滑运动系数 (Smoothing Factor, larger = faster)")]
    [Range(0.1f, 50f)]
    public float smoothFactor = 10f;

    [Header("Debug Info")]
    [SerializeField] private bool isPinching;
    [SerializeField] private Vector3 currentHandPos;
    [SerializeField] private Vector3 targetPos;

    void Start()
    {
        // 如果未设置初始位置，自动使用当前物体位置作为初始位置（可选，根据需求）
        // 这里按照用户要求"物体实际位置需叠加初始位置"，理解为 initialPosition 是一个基准点
        // 如果用户希望以物体当前编辑器位置为基准，可以在编辑器中手动设置 initialPosition
        // 或者在这里初始化：
        // if (initialPosition == Vector3.zero) initialPosition = transform.position;
    }

    void Update()
    {
        // 检查数据管理器单例是否存在
        if (EyeTrackingDataManager.Instance == null) return;

        // 获取手部数据
        isPinching = EyeTrackingDataManager.Instance.IsPinching;
        currentHandPos = EyeTrackingDataManager.Instance.HandData;

        // 仅在捏起状态下更新位置
        if (isPinching)
        {
            // 计算目标位置
            // 映射公式：目标位置 = 初始位置 + (手部坐标 * 缩放比例)
            // 使用 Vector3.Scale 进行逐分量相乘
            targetPos = initialPosition + Vector3.Scale(currentHandPos, scale);

            // 平滑移动
            // 使用 Lerp 插值实现平滑，smoothFactor 控制速度
            transform.position = Vector3.Lerp(transform.position, targetPos, smoothFactor * Time.deltaTime);
        }
    }
}
