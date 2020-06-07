import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd import Function



__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.gradientdomain_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2), #2 classes: Source or Target
        )

    def forward(self, x, alpha=None):
      features = self.features(x)
    
      # If we pass alpha, we can assume we are training the discriminator
      if alpha is not None:
          # gradient reversal layer (backward gradients will be reversed)
          features = features.view(-1, 256 * 6 * 6) 
          reverse_features = ReverseLayerF.apply(features, alpha)
          # Flatten the features:
          discriminator_output = torch.flatten(reverse_features, 1)
          discriminator_output = self.gradientdomain_classifier(discriminator_output)
          return discriminator_output
      # If we don't pass alpha, we assume we are training with supervision
      else:
          # do something else
          class_outputs = self.avgpool(features)
          # Flatten the features:
          class_outputs = torch.flatten(class_outputs,1)
          class_outputs = self.classifier(class_outputs)
          return class_outputs

def dann(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        
        model.gradientdomain_classifier[1].weight.data, model.gradientdomain_classifier[1].bias.data = model.classifier[1].weight.data, model.classifier[1].bias.data
    return model
