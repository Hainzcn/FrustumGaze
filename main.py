import multiprocessing
import sys
from modules.pipeline import FrustumGazePipeline

def main():
    # 启用 multiprocessing 支持
    multiprocessing.freeze_support()
    
    app = FrustumGazePipeline()
    
    try:
        app.run()
    except KeyboardInterrupt:
        pass # run() 方法内部已经处理了 KeyboardInterrupt，这里再次捕获是为了防止初始化阶段的中断
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
    finally:
        # 确保在任何情况下尝试停止
        if hasattr(app, 'stop'):
            app.stop()

if __name__ == "__main__":
    main()
