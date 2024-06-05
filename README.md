# 🦖 DINO Explorer 🦖

DINO Explorer is a powerful tool designed to explore and visualize DINOv2 embeddings. Given a list of folders containing images, DINO Explorer extracts their DINO embeddings and creates an interactive visualization using Voxel51.

## 🚀 Usage

> **Note:** Input folders must be separated by spaces.

* To create a t-SNE visualization, use the following command:

```bash
diex <folder 1> .. <folder N>
```
> Diex uses UMAP for dimension reduction by default.
 
* For t-SNE or PCA visualizations, use the `--m` option followed by `tsne` or `pca`:

```bash
diex <folder 1> .. <folder N> --m tsne
```

* To host the visualization on a specific port, use the `--p` option:

```bash
diex <folder 1> .. <folder N> --p <port>
```

* To set a specific GPU device, use the `--d` option:

```bash
diex <folder 1> .. <folder N> --d <gpu number>
```

After running the command, go to the `add` section, select `embeddings`, choose the brain key: `img_viz` and voila! You have your visualization.

> **Tip:** For multiple folders, select `Color by` as `tags`.

## 💾 Caching Embeddings

DINO Explorer stores DINO embeddings for all datasets in a `.cache` directory to allow quick loading in subsequent visualizations. To force regeneration of embeddings, use the `--force` or `--f` option.

> **Note:** These embeddings can also be used for other downstream tasks. Load them with `torch.load()`.

## 📖 Examples

1. **NuImages** - A random set of 1000 images from NuImages
    ```bash
    diex nuimages_1000
    ```
    A. **Embeddings:** Interactive 2D visualization of embeddings
    ![Embeddings](https://raw.githubusercontent.com/nityanandmathur/diex/main/assests/embed.png)

    B. **Embeddings to Image Mapping:** Select embeddings to view corresponding images
    ![Mapping](https://raw.githubusercontent.com/nityanandmathur/diex/main/assests/mapping.png)

2. **NuImages × CityScapes** - A random set of 1000 and 600 images from the datasets.
    ```bash
    diex nuimages_1000 cityscapes_1000
    ```
    Clusters for different datasets, each with a different color.
    ![Cluster](https://raw.githubusercontent.com/nityanandmathur/diex/main/assests/multiple.png)

## 🙏 Credits

- Model used: `facebook/dinov2-giant`
- Visualization tool: `Voxel51`

> For any issues encountered while using DINO Explorer, please open an issue on our GitHub [repository](https://github.com/nityanandmathur/diex). We appreciate your feedback and contributions!
