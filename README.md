# CAF-GNN 
This repository is an official PyTorch implementation of ["Towards Fair Graph Neural Networks via Graph Counterfactual"](https://arxiv.org/abs/2307.04937) (CIKM 2023). 

## Recent Updates
We're excited to announce that our repository has recently transitioned to using PyTorch Lightning! This integration not only simplifies the codebase and makes our repository more maintainable but also offers numerous benefits like:
- **Scalability**: Effortlessly scale your models to run on more GPUs, TPUs, or even across multiple machines without changing your code.
- **Flexibility**: PyTorch Lightning's modular design makes your code more organized, easier to understand, and allows you to swap different components without hassle.
- **Reproducibility**: Ensures your experiments are reproducible, and you can keep track of all parameters, metrics, and artifacts with minimal code.
### ⚠️ Warning
While we are thrilled about the transition to PyTorch Lightning, it’s important to note that the transfer from the original codebase is not thoroughly verified and may contain small mistakes or inconsistencies. We appreciate your understanding and encourage users to raise any issues or discrepancies found, ensuring continual improvement and refinement of the codebase. Feel free to open an issue for any problems or suggestions you might have!

![Figure](https://github.com/TimeLovercc/CAF-GNN/blob/main/assets/visual.pdf)

## Installation
To set up the necessary environment, follow the steps below:

1. **Clone the repository:**
   ```
   git clone git@github.com:TimeLovercc/CAF-GNN.git
   cd CAF-GNN
   ```

2. **Project Setup:**
   Setting up this project involves a few comprehensive steps. Here's how to get everything ready:

   - **Review and Run Setup Script:**
     It's important to first review the `scripts/setup.sh` script as it includes several setup procedures, some of which might not be necessary for every user, such as the installation of Anaconda. Open the script in your preferred text editor, and feel free to comment out or modify any sections not applicable to your setup. After tailoring the script to your needs, execute it by running:
     ```
     bash scripts/setup.sh
     ```
     This command will take care of installing all the necessary packages and libraries required for the project, according to your adjustments.

   - **Install PyTorch Geometric:**
     Following the base setup, you'll need to install PyTorch Geometric. Detailed instructions for this process are provided [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

   - **Set up Weights & Biases (wandb):**
     Our project utilizes Weights & Biases for efficient experiment tracking, insightful visualization, and convenient data management. Set it up with:
     ```
     pip install wandb
     ```
     If you're not already using wandb, you'll need to create an account. You can log in via the command line using:
     ```
     wandb login
     ```

   These steps will ensure your environment is fully configured and ready for running experiments with our project.


## Usage
After completing the installation steps, you're set to start experimenting with the project. You can initiate the process using the following command:
```
bash scripts/try.sh
```

This script runs the main application with default or pre-specified parameters. However, if you want to customize the execution, you can directly use the command line as follows:
```
CUDA_VISIBLE_DEVICES=[GPU_ID] python ./src/main.py --dataset_name [DATASET] --model_name [MODEL] --seed [SEED] [--no_train]
```
- Replace `[GPU_ID]` with the ID of the GPU you want to use.
- `[DATASET]` can be one of the following: `german`, `credit`, `bail`.
- `[MODEL]` can be one of the following: `gcn`, `sage`, `caf`.
- `[SEED]` is an integer for the random seed.
- Adding `--no_train` is optional and it tells the system to skip the training phase and instead load the latest trained base model.

For instance, to run the application on the 'bail' dataset using the 'sage' model with a specific seed, you would use:
```
CUDA_VISIBLE_DEVICES=2 python ./src/main.py --dataset_name german --model_name sage --seed 2
```

Please ensure you navigate to the project's root directory before initiating the script or command line execution.

For more detailed information or custom configurations, you might want to refer to additional documentation or explore the `./src/main.py` script to understand all available options and how they impact the model's behavior and performance.

## Explore with Jupyter Notebooks

Dive into interactive experimentation and exploration with our Jupyter Notebooks in the `notebooks/example.ipynb`. This illustrative guide facilitates a hands-on approach to data visualization, model training, and analysis within our project's ecosystem. Navigate through each section, from setting up and exploring data to model training and analysis, all while enjoying the flexibility to modify parameters and configurations for personalized experiments. Your journey through model interactions and data insights is enhanced with the capability to run cells, observe real-time results, and even create your own experiment scenarios. Whether you're analyzing, sharing findings, or contributing your own experiments and visualizations, our notebooks provide a comprehensive and interactive platform to enhance your experience with our project. We use `german` dataset and `caf` method as example.

## Data Acquisition
This project relies on specific datasets that need to be prepared before running the models. The datasets can be obtained from an external repository and set up as follows:

1. **Download Data:**
   The datasets are available on [this GitHub repository](https://github.com/chirag126/nifty). Navigate to the repository and download the necessary files. You will need all the `.csv` and `.txt` files for each dataset.

2. **Prepare Data Directory:**
   Within the `CAF-GNN` project, create a directory structure to store the data. If it doesn't already exist, create a `data` folder at the root, and within it, create a subfolder for each dataset you're using (e.g., `data/dataset_name`). Each of these dataset folders should have a `raw` subfolder.

3. **Place Data Files:**
   Copy all the `.csv` and `.txt` files you downloaded for each dataset into the corresponding `data/dataset_name/raw` folder in your local `CAF-GNN` project.

Here's a command-line sequence to make the process clearer:

```sh
# Navigate to your CAF-GNN directory (replace with your actual directory path)
cd path/to/CAF-GNN

# Create a data directory and a subdirectory for your dataset (replace 'dataset_name' with your actual dataset's name)
mkdir -p data/dataset_name/raw

# Now, manually copy the dataset files into the newly created 'raw' folder
```

Please repeat the process for each dataset you plan to use. Ensure the files are correctly placed so the project's scripts can access and process them.

After setting up the data, proceed with the usage instructions as described in the Usage section.


## Contributing
We welcome contributions that improve the code, documentation, or other aspects of the project. If you're interested in contributing, please start by discussing the change you wish to make via an issue. Afterward, you can make your changes and create a pull request.

Please ensure to update tests as appropriate and maintain the quality of the codebase.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Zhimeng Guo - gzjz07 [at] outlook.com

Project Link: [https://github.com/TimeLovercc/CAF-GNN](https://github.com/TimeLovercc/CAF-GNN)

## Citation
If you find our research useful, please consider citing our work:

```bibtex
@article{guo2023towards,
  title={Towards Fair Graph Neural Networks via Graph Counterfactual},
  author={Guo, Zhimeng and Li, Jialiang and Xiao, Teng and Ma, Yao and Wang, Suhang},
  journal={arXiv preprint arXiv:2307.04937},
  year={2023}
}
```

Thank you for your interest in our project! We hope you find this research useful and look forward to seeing your contributions and discussions.
