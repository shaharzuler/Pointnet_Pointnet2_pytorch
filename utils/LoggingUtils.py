from torch.utils.tensorboard import SummaryWriter

class TensorBoardHandler:
    def __init__(self,experiment_dir):
        tb_dir = experiment_dir.joinpath('TB/')
        tb_dir.mkdir(exist_ok=True)
        self.train_writer = SummaryWriter(tb_dir.joinpath('train/'))
        self.val_writer = SummaryWriter(tb_dir.joinpath('val/'))

    def get_writer(self,phase):
        return self.train_writer if phase == "train" else self.val_writer

    def write_loss(self, phase, loss, epoch):
        writer = self.get_writer(phase)
        writer.add_scalar("Loss/", loss, epoch)

    def write_acc(self, phase, acc, epoch):
        writer = self.get_writer(phase)
        writer.add_scalar("Accuracy/", acc, epoch)


