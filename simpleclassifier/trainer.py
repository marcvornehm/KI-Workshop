from typing import Sequence, Type

import numpy as np
import torch
import torch.utils.data
from IPython.display import clear_output
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from .data import SimpleDataset
from .notebook_utils import in_notebook


class Trainer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def train(self, dataset: SimpleDataset, epochs: int = 20, lr: float = 0.001):
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        self.model.train()
        self.model.to(self.device)

        animate = in_notebook()
        losses = []
        prints = []
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device).to(torch.float32)
                labels = labels.to(self.device).to(torch.float32)

                optimizer.zero_grad()

                outputs = self.model(inputs).squeeze(dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            mean_loss = running_loss / len(dataloader)
            losses.append(mean_loss)
            loss_str = f'Epoche {epoch + 1:2.0f} beendet mit loss {mean_loss:.8f}'
            print(loss_str)
            prints.append(loss_str)

            if animate:
                clear_output(wait=True)
                plt.plot(losses)
                plt.gca().get_xaxis().set_major_locator(MaxNLocator(integer=True))
                plt.xlabel('Epoche')
                plt.ylabel('Loss')
                plt.title('Trainingsloss')
                plt.show()
                for p in prints:
                    print(p)

    def test(self, dataset: SimpleDataset):
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        self.model.eval()
        correct = 0
        total = 0
        fp = []
        fn = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                outputs = (outputs > 0.5).to(torch.float32).squeeze(1)
                total += labels.size(0)
                correct += (outputs == labels).sum().item()

                for i in range(len(outputs)):
                    if outputs[i] != labels[i]:
                        if outputs[i] == 1:
                            fp.append(inputs[i].cpu().numpy())
                        else:
                            fn.append(inputs[i].cpu().numpy())

        print(f'Genauigkeit: {correct / total:.2%}')
        print(f'Falsch-Positive (fälschlicherweise als {dataset.LabelPositive.name} erkannt): {len(fp)}')
        print(f'Falsch-Negative (fälschlicherweise als {dataset.LabelNegative.name} erkannt): {len(fn)}')
        if len(fp) > 0:
            self.plot_images(fp, title='Falsch-Positive')
        if len(fn) > 0:
            self.plot_images(fn, title='Falsch-Negative')

    def plot_images(self, images: Sequence[np.ndarray], title: str | None = None):
        rows = len(images) // 5 + 1
        cols = 5
        _, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2 + 0.5), squeeze=False)
        if title is not None:
            plt.suptitle(title)
        for i in range(len(images)):
            img = images[i]
            img -= img.min()
            img /= img.max()
            img = np.moveaxis(img, 0, 2)
            row = i // cols
            col = i % cols
            axs[row, col].imshow(img, cmap='gray')
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            axs[row, col].axis('off')
        plt.tight_layout()
        plt.show()


def reset_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def train(network: torch.nn.Module, dataset: SimpleDataset, epochs: int = 30, lr: float = 0.001, gpu: bool = True):
    network.apply(reset_weights)
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
    trainer = Trainer(network, device)
    trainer.train(dataset=dataset, epochs=epochs, lr=lr)


def test(network: torch.nn.Module, dataset_type: Type):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(network, device)
    dataset = dataset_type(split='test')
    trainer.test(dataset)
