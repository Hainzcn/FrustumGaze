using UnityEngine;
using System.Diagnostics;
using System.IO;

/// <summary>
/// 用于在 Unity 运行时自动启动 Python 服务端脚本
/// 挂载到场景中的任意 GameObject (建议与 EyeTrackingDataManager 挂在一起)
/// </summary>
public class PythonLauncher : MonoBehaviour
{
    [Header("Python Settings")]
    [Tooltip("Python 解释器路径。默认为 'python' (需在环境变量中)。也可指定绝对路径如 'C:/Python39/python.exe'")]
    public string pythonExecutable = "python";
    
    [Tooltip("main.py 路径。可以是相对项目根目录的路径，也可以是绝对路径。")]
    public string scriptPath = "main.py";

    [Tooltip("工作目录。默认为 '.' (自动设为脚本所在目录或项目根目录)。")]
    public string workingDirectory = ".";

    [Tooltip("传递给脚本的命令行参数")]
    public string arguments = "";

    [Header("Runtime Options")]
    [Tooltip("是否显示 Python 控制台窗口")]
    public bool showConsole = true;
    
    [Tooltip("是否在 Start() 时自动启动")]
    public bool launchOnStart = true;
    
    [Tooltip("是否在退出或销毁时自动关闭 Python 进程")]
    public bool killOnQuit = true;

    private Process _process;

    void Start()
    {
        if (launchOnStart)
        {
            Launch();
        }
    }

    public void Launch()
    {
        if (_process != null && !_process.HasExited)
        {
            UnityEngine.Debug.LogWarning("[PythonLauncher] 服务端已在运行中，无需重复启动。");
            return;
        }

        string finalScriptPath = scriptPath;
        string finalWorkingDir = workingDirectory;
        string repoRoot = FindRepoRoot();

        // 路径解析逻辑
        if (!Path.IsPathRooted(finalScriptPath))
        {
            // 如果是相对路径，尝试基于 repoRoot 解析
            if (!string.IsNullOrEmpty(repoRoot))
            {
                finalScriptPath = Path.Combine(repoRoot, finalScriptPath);
                
                // 如果工作目录是默认值 "."，则设为 repoRoot
                if (finalWorkingDir == ".")
                {
                    finalWorkingDir = repoRoot;
                }
                else if (!Path.IsPathRooted(finalWorkingDir))
                {
                    finalWorkingDir = Path.Combine(repoRoot, finalWorkingDir);
                }
            }
            else
            {
                // 无法找到 repoRoot，尝试基于 Application.dataPath 的上级
                // 这是一个回退策略
                finalScriptPath = Path.GetFullPath(Path.Combine(Application.dataPath, "../", finalScriptPath));
            }
        }

        // 最终检查文件是否存在
        if (!File.Exists(finalScriptPath))
        {
            UnityEngine.Debug.LogError($"[PythonLauncher] 找不到脚本文件: {finalScriptPath}\n请检查路径设置或确保 Unity 项目在 FrustumGaze 目录内。");
            return;
        }

        // 确定工作目录
        if (finalWorkingDir == "." || string.IsNullOrEmpty(finalWorkingDir))
        {
            finalWorkingDir = Path.GetDirectoryName(finalScriptPath);
        }

        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.FileName = pythonExecutable;
        startInfo.Arguments = $"\"{finalScriptPath}\" {arguments}";
        startInfo.WorkingDirectory = finalWorkingDir;
        
        startInfo.UseShellExecute = showConsole;
        startInfo.CreateNoWindow = !showConsole;

        // 如果不显示窗口，则重定向输出以便在 Unity Console 查看 (仅限 UseShellExecute = false)
        if (!showConsole)
        {
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.UseShellExecute = false; 
        }

        try
        {
            _process = Process.Start(startInfo);
            UnityEngine.Debug.Log($"[PythonLauncher] Python 服务端已启动 (PID: {_process.Id})\nScript: {finalScriptPath}");

            if (!showConsole)
            {
                // 异步读取输出
                _process.OutputDataReceived += (sender, args) => { if (args.Data != null) UnityEngine.Debug.Log($"[Py Server]: {args.Data}"); };
                _process.ErrorDataReceived += (sender, args) => { if (args.Data != null) UnityEngine.Debug.LogError($"[Py Error]: {args.Data}"); };
                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();
            }
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError($"[PythonLauncher] 启动失败: {e.Message}");
        }
    }

    /// <summary>
    /// 查找包含 main.py 的项目根目录 (从 Assets 向上搜索)
    /// </summary>
    private string FindRepoRoot()
    {
        DirectoryInfo dir = new DirectoryInfo(Application.dataPath);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "main.py")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return null;
    }

    void OnApplicationQuit()
    {
        KillProcess();
    }
    
    void OnDestroy()
    {
        KillProcess();
    }

    private void KillProcess()
    {
        if (killOnQuit && _process != null && !_process.HasExited)
        {
            try 
            {
                _process.Kill();
                UnityEngine.Debug.Log("[PythonLauncher] Python 服务端已关闭。");
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogError($"[PythonLauncher] 关闭进程失败: {e.Message}");
            }
            finally
            {
                _process.Dispose();
                _process = null;
            }
        }
    }
}