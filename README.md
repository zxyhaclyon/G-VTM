# G-VTM

## Title
G-VTM: A Multimodal Vision-Trajectory Model for Generalized Vehicle Trajectory Prediction

## Datasets
We use three real-world datasets: [SinD-Tianjin](https://github.com/SOTIF-AVLab/SinD/tree/main), [InD-Bendplatz](https://www.ind-dataset.com/), and [RounD-Neuweiler](https://levelxdata.com/round-dataset/). The SinD-Tianjin dataset is collected at a signalized intersection in Tianjin, China, recorded at 10 Hz. The InD-Bendplatz dataset is recorded at an unsignalized intersection in Germany at 25 Hz. The RounD-Neuweiler dataset contains trajectories collected from a roundabout in Germany with the same sampling frequency as InD. Please download the dataset to the `raw_data` directory.

## Preprocessing

Assuming that you have been granted access to any of the above-mentioned datasets, proceed by moving the unzipped content (folder) into the folder named `SinD_data/tianjin`, `inD_data/location1`, or `rounD_data/location0` under the `raw_data` directory. 
Download raw datasets from the following sources:
    - <a href='https://pan.baidu.com/s/1LRbTJqnqlz3npOkRqKNaKw?pwd=hzg4'>BaiduNetDisk</a> code: `hzg4`

Methods of preprocessing are contained within Python scripts under the `data_process` directory. Executing them may be done from a terminal or IDE of choice **(from within this project folder)**, for example: 
```bash
python SinD_preprocess.py
python inD_preprocess.py
python RounD_preprocess.py
```
The output of the preprocessing scripts will be sent to a sub-folder with the name of the data set within the `./data` folder in this project.

## Usage
Download the pretrained [vit-base-patch16-224](https://huggingface.co/models) from Hugging Face to the `VLM_Model` directory.
Each dataset corresponds to a configuration file that defines the model, data, and training parameters under the `config` directory.

```bash
python main.py --config ./config/VTM_SinD_Tianjin.conf
```



