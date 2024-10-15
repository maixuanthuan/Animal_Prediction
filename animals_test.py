import numpy as np
import torch
import os
import argparse
from animals_models import CNN
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
def get_agrs():
    parser = argparse.ArgumentParser(description="Animals classifier")
    parser.add_argument('-s', '--image_size', type=int, default=224)
    parser.add_argument('-i', '--image_path', type=str, default="download.jpeg")

    parser.add_argument('-c', '--checkpoint_path', type=str, default="trained_models/best.pt")
    args = parser.parse_args()
    return args

def test(args):
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(categories)).to(device)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("A checkpoint must be provide")
        exit(0)

    if not args.image_path:
        print("A image must be provide")
        exit(0)

    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2,0,1))[None, :, :, :]
    image = image / 255
    # image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(device).float()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        prediction = model(image)
    probs = softmax(prediction)
    max_value, max_index = torch.max(probs, dim=1)
    print("This image is about {} with probality of {}".format(categories[max_index[0].item()], max_value[0].item()))

    fig, ax = plt.subplots()
    ax.bar(categories, probs[0].cpu().numpy())
    ax.set_xlabel("Animal")
    ax.set_ylabel("Probability")
    ax.set_title(categories[max_index])
    plt.show()


if __name__ == '__main__':
    args = get_agrs()
    test(args)
