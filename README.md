# Image Classification Workshop
This material was created for and used in the context of a workshop for 11th graders on AI and Medicine (held in German).

## Teachable Machine
As an introduction to image classification, [Google's Teachable Machine](https://teachablemachine.withgoogle.com/train/image) was used with [this dataset with images of cats and dogs](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification).
See [here](./KatzenVsHunde.md) for further instructions.

## Jupyter Notebook
In a second step, [this Jupyter Notebook](./Workshop_Klassifizierung.ipynb) was used with three datasets to choose from:
1. [Cats vs. Dogs](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification) (same as above)
2. [Brain Tumor Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
3. Cardiac Views (Long axis vs. Short axis). This dataset was created from data from the [Cardiac Atlas Project](https://www.cardiacatlas.org/sunnybrook-cardiac-data/)

The data should be placed in the following structure
```
.
└── data/
    ├── braintumor/
    │   ├── test/
    │   │   ├── notumor/
    │   │   │   └── ... (image files)
    │   │   └── tumor/
    │   │       └── ... (image files)
    │   └── train/
    │       ├── notumor/
    │       │   └── ... (image files)
    │       └── tumor/
    │           └── ... (image files)
    ├── cardiacview/
    │   ├── test/
    │   │   ├── lax/
    │   │   │   └── ... (image files)
    │   │   └── sax/
    │   │       └── ... (image files)
    │   └── train/
    │       ├── lax/
    │       │   └── ... (image files)
    │       └── sax/
    │           └── ... (image files)
    └── catsvsdogs/
        ├── test/
        │   ├── cats/
        │   │   └── ... (image files)
        │   └── dogs/
        │       └── ... (image files)
        └── train/
            ├── cats/
            │   └── ... (image files)
            └── dogs/
                └── ... (image files)
```
