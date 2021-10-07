import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from PIL import Image
import numpy as np
import typing


class BinaryAccuracy:
    def __init__(
            self,
            logits: bool = True,
            reduction: typing.Callable[
                [
                    torch.Tensor,
                ],
                torch.Tensor,
            ] = torch.mean,
    ):
        self.logits = logits
        if logits:
            self.threshold = 0
        else:
            self.threshold = 0.5

        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        return self.reduction(((y_pred > self.threshold) == y_true.bool()).float())


stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
defect_ds = ImageFolder("data/defect/",
                        transform=tt.Compose(
                            [tt.Resize(128),
                             tt.CenterCrop(128),
                             tt.ToTensor(),
                             tt.Normalize(*stats)])
                        )
defect_dl = DataLoader(defect_ds, 128, shuffle=False, num_workers=3, pin_memory=True)


disc = torch.load("./discriminator.pt", map_location=torch.device('cpu'))
img = Image.open("./data/dice/0/200.jpg").convert('RGB')
img_tensor = transforms.ToTensor()(img).unsqueeze_(0)

defect_img = Image.open("defect_crop.jpg").convert('RGB')
defect_tensor = transforms.ToTensor()(defect_img).unsqueeze_(0)

if __name__ == "__main__":
    torch.set_printoptions(precision=10, sci_mode=False)
    out = disc(img_tensor)
    print(out)
    out = disc(defect_tensor)
    print(out)
    print(img_tensor)
