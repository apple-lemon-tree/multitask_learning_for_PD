import torch


class EarlyStopping:
    """
    一个简单的早停机制实现。

    参数:
        patience (int): 在验证损失没有改善的情况下，等待的 epochs 数量。
        verbose (bool): 如果为 True，会在早停时打印一条消息。
        delta (float): 衡量损失改善的最小变化量，小于此值则认为没有改善。
        path (str): 用于保存最佳模型权重的路径。
    """

    def __init__(self,patience=5, verbose=True, delta=1e-4, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.update = False
        # self.best_personality = None
        # self.best_emotion = None

    def __call__(self, score,epoch, checkpoint):

        accuracy = score
        if self.best_score is None:
            self.save_checkpoint(accuracy, checkpoint)
            self.best_score = accuracy
            self.update = True
        elif accuracy <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'早停epoch计数: {self.counter}/{self.patience}')
                self.update = False
            if self.counter >= self.patience:
                self.update = False
                self.early_stop = True
        else:
            self.save_checkpoint(accuracy, checkpoint)
            self.best_score = accuracy
            self.update = True
            self.counter = 0

    def save_checkpoint(self, accuracy, checkpoint):

        if self.verbose:
            print(f'人格准确率+情感准确率 ({self.best_score:.6f} --> {accuracy:.6f}). 保存模型ing...')
        if checkpoint == None:
            pass
        else:
            torch.save(checkpoint, self.path)