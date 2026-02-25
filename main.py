import multiprocessing
from modules.pipeline import FrustumGazePipeline

def main():
    # 启用 multiprocessing 支持
    multiprocessing.freeze_support()
    
    app = FrustumGazePipeline()
    app.run()

if __name__ == "__main__":
    main()
