from models.convnext import ConvNeXtPlus
from torchinfo import summary

if __name__ == '__main__':
    model = ConvNeXtPlus()
    summary(model,input_size=(1,3,224,224),depth=4)