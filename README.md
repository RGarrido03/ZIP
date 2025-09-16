# ZIP: Scalable Crowd Counting via Zero-Inflated Poisson Modeling

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-shanghaitech-a)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-a?p=ebc-zip-improving-blockwise-crowd-counting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-shanghaitech-b)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-b?p=ebc-zip-improving-blockwise-crowd-counting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-ucf-qnrf)](https://paperswithcode.com/sota/crowd-counting-on-ucf-qnrf?p=ebc-zip-improving-blockwise-crowd-counting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ebc-zip-improving-blockwise-crowd-counting/crowd-counting-on-nwpu-crowd-val)](https://paperswithcode.com/sota/crowd-counting-on-nwpu-crowd-val?p=ebc-zip-improving-blockwise-crowd-counting)

<a href="https://www.buymeacoffee.com/yiming.ma" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

The official implementation of the paper [*ZIP: Scalable Crowd Counting via Zero-Inflated Poisson Modeling*](https://arxiv.org/pdf/2506.19955).

🤗 Try our models live! Check out the interactive demo on our [HuggingFace Space](https://huggingface.co/spaces/Yiming-M/ZIP).

## Reults

| **Variants** | **Size (M)** | **GFLOPS (on HD)** | **SHA (MAE)** | **SHA (RMSE)** | **SHA (NAE, %)** | **SHB (MAE)** | **SHB (RMSE)** | **SHB (NAE, %)** | **QNRF (MAE)** | **QNRF (RMSE)** | **QNRF (NAE, %)** | **NWPU-Val (MAE)** | **NWPU-Val (RMSE)** | **NWPU-Val (NAE, %)** | **NWPU-Test (MAE)** | **NWPU-Test (RMSE)** | **NWPU-Test (NAE, %)** |
| ------------ | ------------ | ------------------ | ------------- | -------------- | ---------------- | ------------- | -------------- | ---------------- | -------------- | --------------- | ----------------- | ------------------ | ------------------- | --------------------- | ------------------- | -------------------- | ---------------------- |
| -P (Pico)    | 0.81         | 6.46               | 71.18         | 109.60         | 16.69            | 8.23          | 12.62          | 6.98             | 96.29          | 161.82          | 14.40             | 66.94              | 223.52              | 14.97                 | 79.91               | 327.17               | 21.42                  |
| -N (Nano)    | 3.36         | 24.73              | 58.86         | 94.63          | 14.15            | 7.74          | 12.14          | 6.33             | 86.46          | 147.64          | 12.60             | 56.27              | 292.53              | 14.06                 | 75.03               | 334.54               | 17.59                  |
| -T (Tiny)    | 10.53        | 61.39              | 56.36         | 86.09          | 13.26            | 6.67          | 9.90           | 5.52             | 76.02          | 129.40          | 11.10             | 46.74              | 145.32              | 11.31                 | 66.43               | 323.27               | 13.45                  |
| -S (Small)   | 33.60        | 242.43             | 55.17         | 88.99          | 11.97            | 5.83          | 9.21           | 4.58             | 73.32          | 125.09          | 10.40             | 31.66              | 77.11               | 9.61                  | 62.89               | 309.02               | 12.09                  |
| -B (Base)    | 105.60       | 800.99             | 47.81         | 75.04          | 11.06            | 5.51          | 8.63           | 4.48             | 69.46          | 121.88          | 10.18             | 28.26              | 64.84               | 9.20                  | 60.09               | 298.95               | 10.44                  |

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download Processed Datasets

- **ShanghaiTech A**: [sha.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/sha.zip)
- **ShanghaiTech B**: [shb.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/shb.zip)
- **UCF-QNRF**: [qnrf.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/qnrf.zip), [qnrf.z01](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/qnrf.z01)
- **NWPU-Crowd**: [nwpu.zip](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.zip), [nwpu.z01](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z01), [nwpu.z02](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z02), [nwpu.z03](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z03), [nwpu.z04](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z04), [nwpu.z05](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z05), [nwpu.z06](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z06), [nwpu.z07](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z07), [nwpu.z08](https://github.com/Yiming-M/EBC-ZIP/releases/download/dataset/nwpu.z08)

To unzip splitted `.zip` files, 7-Zip is recommended. You can use the following command to install 7-Zip and unzip the dataset:

```bash
sudo apt update
sudo apt install p7zip-full

7z x dataset.zip
```

## Step 3: Run Training

Add the training code to `run.sh` and execute it:

```bash
sh run.sh
```

If you want to use the zero-inflated loss, set either `--reg_loss` or `--aux_loss` to `zipnll`. For example, you can set `--reg_loss zipnll` to use the zero-inflated loss for regression. 

You can use an auxillary loss to improve the performance. For example, you might want to use the pre-defined multi-scale MAE loss by setting `--aux_loss msmae` and `--scales 1 2 4`.

The DMCount loss can also be used together with the zero-inflated loss. For example, you can set `--reg_loss zipnll --aux_loss dmcount` to use both losses.


## Step 4: Test the Model

Use `test.py` or `test.sh` to test the model. You can specify the dataset, weight path, input size, and other parameters.

To generate the predicted counts on NWPU-Crowd Test, you need to use `test_nwpu.py` instead.

To visualize the results, use the `notebooks/model.ipynb` notebook.

Trained weights are also provided:
- [**ShanghaiTech A**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_sha)
- [**ShanghaiTech B**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_shb)
- [**UCF-QNRF**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_qnrf)
- [**NWPU-Crowd**](https://github.com/Yiming-M/EBC-ZIP/releases/tag/weights_nwpu)

Make sure to use the processed datasets and the exact commands pre-defined in `test.sh` to reproduce the same results.

## Step 5: Visualize the Results

Use the `notebooks/model.ipynb` notebook to visualize the results.
