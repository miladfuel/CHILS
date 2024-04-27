# CHiLS

## Practical Example of CHILS:

Please refer to the original author's README after reviewing the following notes about my contributions and changes:

### What have I done?
- **a.** Installed Linux, conda, CUDA, and PyTorch, and set OPENAI_API_KEY.
- **b.** Updated the conda environment.
- **c.** Made all .py files compatible with updated packages.
- **d.** Selected specific folders of our dataset (e.g., from food-101, I chose 10 folders such as apple) to expedite code execution. I renamed the folder names to lowercase. If there is a space between words, I replaced it with an underscore. However, in the label set JSON file, we do not use underscores (e.g., 'African crowned crane').
- **e.** Created a .py file to generate a meta folder, including 5 new files. This script: 1- Makes image names unique, 2- Converts images to RGB, 3- Generates a meta folder. The script is located at `chils/data/food-101`. To run it, activate the conda environment and execute `python create_meta.py` in the terminal.
- **f.** Adjusted the label set JSON to match the number of dataset folders used, renamed it to `food-101-10.json`, and moved it to the `label_sets` folder. This naming convention ensures compatibility when specifying the label set size in subsequent scripts.
- **g.** Executed `run.py` which generates a `ClipViTL14` folder containing an `.npz` file, focusing on the robustness of the model.
- **h.** Ran `zshot.py` using the following command: `python zshot.py --dataset=food-101 --model=ClipViTL14 --experiment=gpt --label-set-size=10 --data-dir=/home/milad/CHILS/data --out-dir=/home/milad/CHILS/outputs`
- **i.** Modified the label set and dataset, re-ran the scripts, and compared outputs.

### Configuration needed:
- **Folder `config`**, file `config.yaml`: Specify the path and architecture.
- **Folder `config/datamodule`**, file `data.yaml`: If your GPU does not support a batch size of 64, change it to 32. Also, change `src/datamodule.py` (you need to change two parameters from 64 to 32).
- **Folder `src`**, file `extract_feats.py`: Change the `base_task` dataset. See `data_utils.py` for options. This parameter can be changed to other datasets given by constant.py dataset array.
=======
I have also added the `sea life` dataset, renamed to `food-101`, to demonstrate using external datasets outside of the standard PyTorch datasets.


### Quick Start:
If you want to quickly run the project without extensive setup:
1. Do not modify the setup instructions detailed above.
2. Download your dataset, and place image folders similar to my structure under `CHILS/data/food-101/images`.
3. Execute `create_meta.py` as detailed above.
4. Create a label set JSON file like `food-101-10.json` in the `Label_sets` folder and replace the existing file.
5. Run `run.py`.
6. Execute `zshot.py` with the specified command, adjusting the path to match your system.

### Customizing for Other Datasets:
If you wish to use a different dataset:
- **Configurations**: Adjustments are required in the `config` and `datamodule` directories. Specifically, modify `config.yaml` and `data.yaml` to reflect your system's capabilities and dataset paths.
- **Code Modifications**: In the `src` directory, wherever `food-101` is referenced, replace it with your dataset's name. For example, modify `extract_feats.py` and `data_utils.py` accordingly.

For datasets not included in the PyTorch ecosystem, update `src/data_utils.py` to include a download option:

```python
elif dataset.lower() == "eurosat":
    data = esat_idx(data_dir, transform=transform, download=True)


After downloading, manually extract the data if not automatically done.

### System Requirements:
- **Operating System**: Linux Ubuntu 22.04.
- **Hardware**: NVIDIA GPU (tested with NVIDIA RTX 3050).
- **Software**: CUDA 11.8, conda. Ensure to use the `environment.yml` file provided for setting up the conda environment.

### Running the Code:
To run the scripts:
1. Activate the environment with `conda activate clip-hierarchy`.
2. Navigate to the CHILS directory (`cd CHILS`).
3. Execute `python run.py` to test your dataset and create output files and directories.

### Output Interpretation:
`zshot.py` will generate detailed performance reports such as:
- **Superclass Accuracy**: Reflects the model's ability to correctly identify broad categories.
- **CHiLSNoRW**: Measures accuracy without hierarchical adjustments, sensitive to detailed subclass predictions.
- **CHiLS**: Demonstrates the effectiveness of hierarchical adjustments in maintaining accuracy despite potential errors in subclass predictions.

These metrics illustrate the resilience and adaptability of the model across various levels of classification detail.
```

After downloading, if program did not extract it, you need to extract it manually (right click and choose extract here).

 ------


## Overall:
**Note:** I installed Linux Ubuntu 22.04 and created a conda environment using `environment.yml` from this GitHub repository after cloning. My laptop has an NVIDIA RTX 3050 GPU. Therefore, you need to have an NVIDIA GPU because this project uses CUDA. When you install an updated version of Ubuntu, it will automatically install your graphics driver. So, you just need to create an environment on conda using my `environment.yml` file. I used CUDA 11.8, so you can find the syntax from the original website if it does not work.

1. **`run.py`** will test your dataset and create a folder named after the architecture (e.g., `ClipViTL14`) in the working directory. It will contain a file in `.npz` format. Additionally, `run.py` will create a checkpoint in the `outputs` folder.
To run the code this is my steps in terminal:
a) conda activate clip-hierarchy
b) cd CHILS
c) python run.py

2. **`zshot.py`** will create a file like `food-101-ClipViTL14-gpt-10-normal` in the `outputs` folder to show you a report like this:
   - Superclass: 93.87%
   - CHiLSNoRW: 90.83%
   - CHiLS: 93.79%

   Here is an example of the syntax to run `zshot.py`:
   ```
   python zshot.py --dataset=food-101 --model=ClipViTL14 --experiment=gpt --label-set-size=10 --data-dir=/home/milad/CHILS/data --out-dir=/home/milad/CHILS/outputs
   ```

**Note:** I added some print statements in `extract_feats.py` and `data_utils.py`, which can be removed if you do not need them for debugging.

-------------------------------
Reference:
https://github.com/acmi-lab/CHILS/

![CHiLS](fig19.jpeg)

This is the official implementation for [CHiLS: Zero-shot Image Classification with Hierarchical Label Sets](https://arxiv.org/abs/2302.02551). If you find this repository useful or use this code in your research, please cite the following paper: 

> Zachary Novack, Julian McAuley, Zachary Lipton, and Saurabh Garg. Chils: Zero-shot image classification with hierarchical label sets. In International Conference on Machine Learning (ICML), 2023.
```
@inproceedings{novack2023chils,
    title={CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets},
    author={Novack, Zachary and McAuley, Julian and Lipton, Zachary and Garg, Saurabh},
    year={2023},
    booktitle={International Conference on Machine Learning (ICML)}, 
}
```

There are three main steps for recreating the paper results:

1. Setting up the environment and datasets
2. Caching the CLIP-extracted features for each dataset and model
3. Running zero-shot inference


## Setting up the environment and datasets:
All requisite packages can be installed via the `environment.yml` file. For access to GPT-3 through OpenAI, you must have an account and save your access token in the environment variable `OPENAI_API_KEY`.

Besides ImageNet, CIFAR100 and Fashion-MNIST (which can be autoloaded through the `torchvision` API), each dataset can be downloaded through the standard websites for each: [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code), [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html), [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101), [Fruits360](https://www.kaggle.com/datasets/moltean/fruits), [Fashion1M](https://github.com/Cysu/noisy_label), [LSUN-Scene](https://www.yf.io/p/lsun), [ObjectNet](https://objectnet.dev/).
Dataset Notes:
- Both LSUN-Scene and Fashion1M must be configured into the `ImageFolder` format, wherein the directory has named folders for each class, each containing all the images. Due to compute constraints, for LSUN-Scene we use the validation data only and for Fashion1M we use the first two large image folders (i.e. `0` and `1`).

## Caching the CLIP-extracted features for each dataset and model:
Running `run.py` will use the variables specified in `config.yaml` and extract the features of a given dataset and CLIP model. In order to run this, the variable `data_loc` must be changed to the directory where your datasets are held.

## Running zero-shot inference:
Once the features are extracted, you may run `zshot.py` to generate the zero-shot inference results with CHiLS. For example, to generate the results with the GPT-generated label sets (which are provided for reproducibility) on Food-101, the command would be:

```
python zshot.py --dataset=food-101 --model=ClipViTL14 --experiment=gpt --label-set-size=10 --data-dir=[INSERT YOUR PATH HERE]
```

See the `src/constants.py` file for valid inputs for each argument in the command.

=======

