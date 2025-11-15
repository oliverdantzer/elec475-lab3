import typer
from typing import Literal
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def test(eval_func):
    print()

def main(model: Literal["resnet50-fcn"]):
    print(f"Hello {model}")

if __name__ == "__main__":
    typer.run(main)