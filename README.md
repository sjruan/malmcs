# Dynamic Public Resource Allocation based on Human Mobility Prediction

## Paper

If you find our code or dataset useful for your research, please cite our paper:

Sijie Ruan, Jie Bao, Yuxuan Liang, Ruiyuan Li, Tianfu He, Chuishi Meng, Yanhua Li, Yingcai Wu and Yu Zheng. "Dynamic Public Resource Allocation based on Human Mobility Prediction.", ACM IMWUT/UbiComp 2020.

## Requirements

Python 3.6

* numpy==1.14.5
* networkx==2.2
* shapely==1.6.4
* pickle

## Dataset

We organize our dataset into two archives, i.e., `MALMCS_data.zip` and `PREDICTION_data.zip`

1. MALMCS_data.zip

* `frames_20180101_20181101_24.npy`: this is the hourly crowd flows data in Beijing Happy Valley from 01/01/2018 to 01/11/2018 scraped from the [Tencent Heat Map](https://heat.qq.com/). The last month is used for evaluation, and previous months are used for training & validation.

* `pred_all_stresnet_mf4_masked.pkl`: this is the predicted results from the prediction model for evaluation acceleration purpose. In the paper, those results are obtained by training [MF-STN](https://github.com/panzheyi/MF-STN).

2. PREDICTION_data.zip

This archive provides some external factors for crowd flow prediction, which can be used to train the crowd flow prediction model together with `frames_20180101_20181101_24.npy`. This dataset is also a data source for [UrbanFM](https://github.com/yoshall/UrbanFM).
* holiday features: `external/holiday_20180101_20181101_24.npy`
* meteorology features: `external/mete_cy_20180101_20181101_24.npy`
* ticket price features: `external/price_20180101_20181101_24.npy` 
* time of day features: `external/tod_20180101_20181101_24.npy`


## Usage

Tunable Parameters
* Service radius `radius`
* Energy limitation `cost_limit` 
* Number of agents `k`

```
python evaluate.py
```

## License

The code and data are released under the MIT License.