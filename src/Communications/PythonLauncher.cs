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

    [Tooltip("优雅退出等待时间（毫秒），超时后强制结束")]
    public int gracefulShutdownTimeoutMs = 2000;

    private Process _process;
    private bool _isStopping;

    void Start()
    {
        if (launchOnStart)
        {
            Launch();
        }
    }

    public void Launch()
    {
        if (_process != null && _process.HasExited)
        {
            _process.Dispose();
            _process = null;
        }

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
        string extraArguments = string.IsNullOrWhiteSpace(arguments) ? "" : $" {arguments}";
        startInfo.Arguments = $"-u \"{finalScriptPath}\"{extraArguments}";
        startInfo.WorkingDirectory = finalWorkingDir;
        startInfo.UseShellExecute = false;
        startInfo.CreateNoWindow = !showConsole;
        startInfo.RedirectStandardInput = true;
        startInfo.RedirectStandardOutput = !showConsole;
        startInfo.RedirectStandardError = !showConsole;

        try
        {
            _process = Process.Start(startInfo);
            _process.EnableRaisingEvents = true;
            _process.Exited += OnPythonProcessExited;
            UnityEngine.Debug.Log($"[PythonLauncher] Python 服务端已启动 (PID: {_process.Id})\nScript: {finalScriptPath}");

            if (!showConsole)
            {
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
        if (_isStopping)
        {
            return;
        }

        _isStopping = true;
        try
        {
            if (_process == null)
            {
                return;
            }

            if (_process.HasExited)
            {
                _process.Dispose();
                _process = null;
                return;
            }

            if (!killOnQuit)
            {
                return;
            }

            bool exited = false;
            try
            {
                if (_process.CloseMainWindow())
                {
                    exited = _process.WaitForExit(gracefulShutdownTimeoutMs);
                }
            }
            catch {}

            if (!exited && !_process.HasExited)
            {
                _process.Kill();
                _process.WaitForExit(1000);
            }

            UnityEngine.Debug.Log("[PythonLauncher] Python 服务端已关闭。");
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError($"[PythonLauncher] 关闭进程失败: {e.Message}");
        }
        finally
        {
            if (_process != null)
            {
                _process.Dispose();
                _process = null;
            }
            _isStopping = false;
        }
    }

    private void OnPythonProcessExited(object sender, System.EventArgs e)
    {
        UnityEngine.Debug.Log("[PythonLauncher] Python 服务端进程已退出。");
    }
}
