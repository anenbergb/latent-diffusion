import torch
from torchmetrics.image import FrechetInceptionDistance, InceptionScore


class ImageQualityMetrics:
    """
    Metrics
        FID (Frechet Inception Distance) https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
            - compares the final 2048 dim features that capture semantic content
        sFID
            - same as FID score, but compares the early or intermediate InceptionV3 features that are more sensitive to
            style, texture, and low-level appearance. We use the 768-dim features. 
        IS (Inception Score) https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html

    """
    def __init__(self, device: torch.device = torch.device("cuda")):
        self.FID = FrechetInceptionDistance(feature=2048).to(device) # default
        self.sFID = FrechetInceptionDistance(feature=768).to(device) # sFID
        self.IS = InceptionScore().to(device)
        self.device = device

    def update(self, real=None, fake=None):
        if real is not None:
            real = real.to(self.device)
            self.FID.update(real, real=True)
            self.sFID.update(real, real=True)

        if fake is not None:
            fake = fake.to(self.device)
            self.FID.update(fake, real=False)
            self.sFID.update(fake, real=False)
            self.IS.update(fake)

    def compute(self):
        inception_score = self.IS.compute()
        return {
            "FID": self.FID.compute().item(),
            "sFID": self.sFID.compute().item(),
            "Inception_Score_mean": inception_score[0].item(),
            "Inception_Score_std": inception_score[1].item(),
        }
