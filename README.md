# CHiLS


## Practical Example of CHILS:

Please read the original author's README after my note below:

I have included the `food-101` dataset in the codes and updated it with new packages.

### Configuration needed:
- **Folder `config`**, file `config.yaml`: Specify the path and architecture.
- **Folder `config/datamodule`**, file `data.yaml`: If your GPU does not support a batch size of 64, change it to 32. Also, change `src/datamodule.py` (you need to change two parameters from 64 to 32).
- **Folder `src`**, file `extract_feats.py`: Change the `base_task` dataset. See `data_utils.py` for options. This parameter can be changed to other datasets such as `cifar100`, `fruits360`, `eurosat`, `lsun-scene`, `fashion1M`, `imagenet`, etc.

If you do not have the dataset, you can modify `src/data_utils.py` for the related dataset and add `download=True` like this:

```python
elif dataset.lower() == "eurosat":
    data = esat_idx(data_dir, transform=transform, download=True)
```

After downloading, you need to extract it manually.


 ------


## Overall:
**Note:** I installed Linux Ubuntu 22.04 and created a conda environment using `environment.yml` from this GitHub repository after cloning. My laptop has an NVIDIA RTX 3050 GPU. Therefore, you need to have an NVIDIA GPU because this project uses CUDA. When you install an updated version of Ubuntu, it will automatically install your graphics driver. So, you just need to create an environment on conda using my `environment.yml` file. I used CUDA 11.8, so you can find the syntax from the original website if it does not work.

1. **`run.py`** will test your dataset and create a folder named after the architecture (e.g., `ClipViTL14`) in the working directory. It will contain a file in `.npz` format. Additionally, `run.py` will create a checkpoint in the `outputs` folder.

2. **`zshot.py`** will create a file like `food-101-ClipViTL14-gpt-10-normal` in the `outputs` folder to show you a report like this:
   - Superclass: 93.87%
   - CHiLSNoRW: 90.83%
   - CHiLS: 93.79%

   Here is an example of the syntax to run `zshot.py`:
   ```
   python zshot.py --dataset=food-101 --model=ClipViTL14 --experiment=gpt --label-set-size=10 --data-dir=/home/milad/CHILS/data --out-dir=/home/milad/CHILS/outputs
   ```

**Note:** I added some print statements in `extract_feats.py` and `data_utils.py`, which can be removed if you do not need them for debugging.

**Note:** In the `data` folder (e.g., `food-101`), we need to have both the `food-101.gz` file and a folder named `food-101`, which is extracted from this gz file and contains an `images` folder. Inside the `images` folder, there should be 101 subfolders, and the labels generated by this program should match these 101 categories. So, we have data, outputs, and specific arch folder, which are not here because they are empty now.


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
