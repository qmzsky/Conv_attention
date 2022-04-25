from models.convnext import ConvNeXtPlus,ConvNeXt
from torchinfo import summary

if __name__ == '__main__':
    model = ConvNeXtPlus()
    summary(model,input_size=(3,3,224,224),depth=4)
