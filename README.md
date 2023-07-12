# hyconditm_paddle

Paper: Hybrid Conditional Deep Inverse Tone Mapping.

[Paper](https://paperswithcode.com/paper/a-new-journey-from-sdrtv-to-hdrtv)

作者: Tong shao等

Paddle 复现版本

## 数据集

https://pan.baidu.com/s/1OSLVoBioyen-zjvLmhbe2g

提取码: 2me9

## 训练模型

链接：https://pan.baidu.com/s/1ehapDcpGIWY3wPSObj44iA?pwd=hh66 
提取码：hh66

## 训练步骤

### train 

```bash
python train.py -opt config/train/train_hyconditm.yml
```

```
多卡仅需
​```bash
python -m paddle.distributed.launch train.py --launcher fleet -opt config_file_path
```

## 测试步骤

```bash
python test.py -opt config/test/test_hyconditm.yml
```

## 复现指标

|        | PSNR  |
| ------ | ----- |
| Paddle | 37.94 |

注：因显存限制，测试结果为测试图片降采样到1080p的结果