from models.convnext import ConvNeXtPlus,ConvNeXt, SwinTransformerBlock, ConvNextAttention
from torchinfo import summary
import torch.nn as nn
import torch


if __name__ == '__main__':
      model = ConvNextAttention(transfer="D:/wlr/python/pth/convnext_tiny_1k_224_ema.pth")
      summary(model,input_size=(3,3,224,224),depth=4)

    # model2 = ConvNeXt()
    # checkpoint = torch.load("D:/wlr/python/pth/convnext_tiny_1k_224_ema.pth")
    # model2.load_state_dict(checkpoint['model'])
    # model2.stages[3][2] = SwinTransformerBlock(dim=768,input_resolution=[7,7],num_heads=12)
    # summary(model2,input_size=(3,3,224,224),depth=4)


    # model3 = ConvNeXtPlus()
    # checkpoint = torch.load("D:/wlr/python/pth/100epoch.pth",map_location="cpu")["model"]
    # del checkpoint["head.weight"]
    # del checkpoint["head.bias"]
    # model3.load_state_dict(checkpoint)
    # summary(model3,input_size=(3,3,224,224),depth=4)