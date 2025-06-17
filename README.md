# Combination Encoder Decoder Tranformers to Summarize Documents with Text Clustering and Topic Modeling

<p>
  <img alt="Encoder Decoder Transformer" src="imgs/enc_dec.png"/>
</p>

[img source](https://medium.com/@minh.hoque/a-comprehensive-overview-of-transformer-based-models-encoders-decoders-and-more-e9bc0644a4e5)

## Project Description

This project was based off of a tutorial from the book, **[The StatQuest Illustrated Guide to Neural Networks and AI](https://www.amazon.com/dp/B0DRS71QVQ)**

The central ideas expressed in this project are illustrated below.

<img src="https://github.com/StatQuest/signa/blob/main/chapter_14/images/text_summarization_overview.png?raw=1" alt="the curse of high dimensionality" width="800" />

Taking several documents based on different topics as inputs, I convert the documents into embeddings with an encoder-only transformer. This encoder-only transformer represents a `Representative AI` (i.e., A Respresentation Learner). These are systems that uses "representations" to process and understand data. In my case, the encoder's role is to take the raw data input and transform it into a compressed, lower-dimensional representation also known as a `latent space`. This compressed information preserves the essential information because the encoder learns toe extract the meaningful features from the data. In my case, by encoding, I am clustering the documents based similar themes, topics, and meaning rather than just keywords or neighboring words.

I reduce the dimensions of the embeddings using Uniform Manifold Approximatio and Projection (`UMAP`), a dimensionality reduction techinique used to visualize high-dimensional data into lower dimensional space, while stil preserving the the datas global strcuture and local relationships. It is particularly helpful for revealing patterns and clusters in compelx datasets.

I cluser the documents with the reduced embeddings using the density-based clustering algorithm Hierarchical Density-Based Spatial Clustering of Applications with Noise (`HDBSCAN`). HDBSCAN, like DBSCAN, identifies clusters based on the density of data points. Areas with high density of points are considered clusters, spareser areas are either noise or boundaries between clusters. Unlike DBSCAN, HDBSCAN automatically determines the number of clusters and can handle clusters of varying densities. This clustering is accomplished because HBSCAN constructs a hierarchy of clusters and then extracts a stable set of clusters using a density-based measures of stability, like Excess of Mass, to select the most persistent and well-defined clusters. Additionally, it is able to identify and leable outliers as noise rather than forcing them into a cluster like DBSCAN.

Once I have each cluster, I give each cluster a title

Next, I use a Decoder-Only LLM, or Generative AI, to give each cluster an excellent title without knowing anything about these documents in advance. It is Generative AI because I am generating new content (here titles) based on learned patterns frm existing data. Decoders take compressed representations of data (in this case generated from an encoder) and generate new meaninful (well, we shall see) output. The job of the decoder is to tak abstract data representations and turn them into something understandable.

Overall, this project is great example of the power of AI and the ability of transformers to aid in exploratory data analysis and how to deal with vast archives that I otherwise would need help with to form associations.

### My Solution

I build a single Jupyter notebook that constructs and encoder and decoder.

---

## Objective

The project contains the key elements:

- `Encoder-only`, Large Language model (LLM) or Representative AI, to discover and earn the most useful fetures from documents,
- `Decoder-only`, or Generative AI, to create new titles from encoded representations,
- `Deep Learning` for neural networks building,
- `Git` (version control),
- `Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)` for density-based clustering,
- `Jupyter` python coded notebooks,
- `Latent Space`, the lower-dimensional respresentation of the input data to the encoder,
- `Long Short-Term Memory (LSTM)` neural network variant of RNN to process sequential data,
- `Natural Language Processing (NLP)`
- `Python` the standard modules,
- `PyTorch` Machine Learning framework to train our deep neural network,
- `Recurrent Neural Network (RNN)` feedback loop neural networks to process sequential data,
- `Tensors` mathematical objects that generalize scalars, vectors, and matrices into higher dimensions. A multi-dimensional array of numbers,
- `TensorBoard` visualization toolkit for TensorFlow that provides tools and visualizations for machine learning experimentation
- `Transformers`, to manipulate data,
- `Uniform Manifold Approximatio and Projection (UMA)`, a dimenstionality reduction technique, and
- `uv` package management including use of `ruff` for linting and formatting

---

## Tech Stack

![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

---

## Getting Started

Here are some instructions to help you set up this project locally.

---

## Installation Steps

The Python version used for this project is `Python 3.12` to be compatible with `PyTorch`.

Follow the requirements for [Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

- Make sure to use python versions `Python 3.12`
- pip version 19.0 or higher for Linux (requires manylinux2014 support) and Windows. pip version 20.3 or higher for macOS.
- Windows Native Requires Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

### Clone the Repo

1. Clone the repo (or download it as a zip file):

   ```bash
   git clone https://github.com/beenlanced/ltsm_project_pytorch.git
   ```

2. Create a virtual environment named `.venv` using `uv` Python version 3.12:

   ```bash
   uv venv --python=3.12
   ```

3. Activate the virtual environment: `.venv`

   On macOs and Linux:

   ```bash
   source .venv/bin/activate #mac
   ```

   On Windows:

   ```bash
    # In cmd.exe
    venv\Scripts\activate.bat
   ```

4. Install packages using `pyproject.toml` or (see special notes section)

   ```bash
   uv pip install -r pyproject.toml
   ```

### Install the Jupyter Notebook(s)

1. **Run the Project**

   - Run the Jupyter Notebook(s) in the Jupyter UI or in VS Code.

---

## Special Notes

- Getting TensorBoard to work with VS code if you are using VS Code

  - Get the [VS Code Extension](https://devblogs.microsoft.com/python/python-in-visual-studio-code-february-2021-release/)

  - [additional reference](https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode)

- To start a TensorBoard session from VS Code:

  - Open the command palette (Ctrl/Cmd + Shift + P)

    - you may need to add tensorboard to your current virtual environment
      - in terminal I used `uv add tensorbard` as I use uv to add modules.
      - Note: all of this should be done for you with this project as all dependencies are in the pyproject.toml file.

  - Search for the command `Python: Launch TensorBoard` and press enter.
  - You will be able to select the folder where your TensorBoard log files are located. By default, the current working directory will be used. Her, I used the `lightning_logs` directory.

    - VS Code will then open a new tab with TensorBoard and its lifecycle will be managed by VS Code as well. This means that to kill the TensorBoard process all you have to do is close the TensorBoard tab.

---

### Final Words

Thanks for visting.

Give the project a star (⭐) if you liked it or if it was helpful to you!

You've `beenlanced`! 😉

---

## Acknowledgements

I would like to extend my gratitude to all the individuals and organizations who helped in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, have been invaluable. Thank you.

Specifically, I would like to acknowledge:

- [Joshua Starmer - StatQuest](https://youtu.be/YCzL96nL7j0). This project was based off of his Illustrated Guide to Neural Networks and AI course, particularly chapter 14's Encoder-Only Transformers which itelf is heavily based on the tutorial in Chapter 5 of _[Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)_ by [Jay Alammar](https://www.linkedin.com/in/jalammar/) and [Maarten Grootendorst](https://www.linkedin.com/in/mgrootendorst/). Thanks!

- [Hema Kalyan Murapaka](https://www.linkedin.com/in/hemakalyan) and [Benito Martin](https://martindatasol.com/blog) for sharing their README.md templates upon which I have derieved my README.md.

- The folks at Astral for their UV [documentation](https://docs.astral.sh/uv/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details
