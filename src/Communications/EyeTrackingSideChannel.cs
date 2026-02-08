using UnityEngine;
using Unity.MLAgents.SideChannels;
using System;

public class EyeTrackingSideChannel : SideChannel
{
    // 定义一个唯一的 UUID，必须与 Python 端一致
    // 使用在线 UUID 生成器生成的示例 UUID
    public const string ChannelId = "621f0a70-4f87-11ea-a6bf-784f4387d1f7";
    
    // 定义一个回调委托，当接收到数据时通知外部
    public Action<float, float, float> OnDataReceived;

    public EyeTrackingSideChannel()
    {
        ChannelId = new Guid(ChannelId);
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // 解析接收到的消息
        // 假设 Python 端发送的是 3 个 float (x, y, z)
        if (msg.ReadableFloatListCount >= 3)
        {
            var floats = msg.ReadFloatList();
            float x = floats[0];
            float y = floats[1];
            float z = floats[2];

            // 调用回调
            OnDataReceived?.Invoke(x, y, z);
        }
    }
}
