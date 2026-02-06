using UnityEngine;
using System.Collections;

public class EyeBlinker : MonoBehaviour
{
    public SkinnedMeshRenderer meshRenderer;
    public int blinkShapeIndex = 0; // 眨眼 BlendShape 的索引
    public float blinkSpeed = 10f;  // 眨眼速度
    public Vector2 blinkIntervalRange = new Vector2(2f, 5f); // 随机眨眼间隔（秒）

    void Start()
    {
        if (meshRenderer == null)
            meshRenderer = GetComponentInChildren<SkinnedMeshRenderer>();

        // 启动眨眼循环
        StartCoroutine(BlinkRoutine());
    }

    IEnumerator BlinkRoutine()
    {
        while (true)
        {
            // 1. 等待一段随机时间
            float waitTime = Random.Range(blinkIntervalRange.x, blinkIntervalRange.y);
            yield return new WaitForSeconds(waitTime);

            // 2. 闭眼
            yield return StartCoroutine(ToggleEye(0, 100));

            // 3. 睁眼
            yield return StartCoroutine(ToggleEye(100, 0));
        }
    }

    IEnumerator ToggleEye(float start, float end)
    {
        float t = 0;
        while (t < 1f)
        {
            t += Time.deltaTime * blinkSpeed;
            float weight = Mathf.Lerp(start, end, t);
            meshRenderer.SetBlendShapeWeight(blinkShapeIndex, weight);
            yield return null;
        }
    }
}
