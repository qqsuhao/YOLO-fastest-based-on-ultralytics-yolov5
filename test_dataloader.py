#! 我加入了测试代码，测试修改后的dataloader，顺带看看数据增强的效果
if __name__ == "__main__":
    from pathlib import Path
    from utils.general import colorstr
    from utils.datasets import create_dataloader
    import os, sys

    FILE = Path(__file__).resolve()         #! 获取当前文件的绝对路径
    ROOT = FILE.parents[0]
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
    RANK = int(os.getenv('RANK', -1))               #! 关于这三个变量的具体含义，可以参考https://blog.csdn.net/hxxjxw/article/details/119606518
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

    train_path = "/home/suhao/paper/yolo/data/train/train.txt"
    imgsz = 320
    batch_size = 8
    WORLD_SIZE = 1
    gs = 32
    single_cls = False
    hyp = ROOT / 'data/hyps/hyp.VOC.yaml'
    rect = False
    image_weights = False
    quad = False
    workers = 8
    cache = "null"


    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,                      #! 加载训练数据
                                              hyp=hyp, augment=True, cache=None if cache == 'val' else cache,
                                              rect=rect, rank=LOCAL_RANK, workers=workers,
                                              image_weights=image_weights, quad=quad,
                                              prefix=colorstr('train: '), shuffle=True)
    