# Shufflenet-in-lung-colon-image-classification
Shufflenet in lung colon image classification

## Project

```
│  .gitignore
│  datalist.py	//暂时没用
│  dataset.py	//用于生成Pytorch Dataset和DataLoader
│  data_tools.py //用于对原始数据进行切分，划分为训练集、测试集和验证集
│  LICENSE
│  main.py		
│  predict.py	//预测器
│  README.md	
│  shufflenet.py //网络架构
│  train.py		 //训练器
│
├─cancer_data	//切分后的数据
├─log			//tensorboard日志
├─lung_colon_image_set //原始数据
└─model			//训练时保存模型
        onnx_final_model.onnx
        train_model_epoch_58_acc_0.9860000014305115.pkl
        train_model_epoch_59_acc_0.9832000136375427.pkl
        train_model_epoch_60_acc_0.979200005531311.pkl
```

环境说明：

- `Python 3.9`
- `Pytorch 1.13.0+cu116`
- `tensorboard 2.11.0 `

效果：

- 训练轮数60轮，验证集准确率97.92%
- 测试集准确率：98.5%
