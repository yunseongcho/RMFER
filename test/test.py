from models.efficientnet import EfficientNet

model = EfficientNet(emotions=7, scale=0.25, self_masking=True, sampling_ratio=0.1)
