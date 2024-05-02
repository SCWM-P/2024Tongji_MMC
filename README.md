# 2024 Tongji Mathmatical Model Competition

## 成员

- 2251*** 彭博
- 2251*** 卞政
- 2251*** 周子睿

本仓库的代码内容为2024.4.28~2024.5.2数模校赛期间对C题的完成与实践代码，可以下载运行：

```bash
git clone https://github.com/SCWM-P/2024Tongji_MMC.git [directory]
```

依赖库的安装涉及torch，可以直接使用requirements.txt完成依赖的安装：

```bash
pip install -r requirements.txt
```

- 主程序为计算所有的算法的值并保存在`tid2013`对应的文件夹下各自的txt文件中，可以与标准的值做比较，`metrics_values`为数据集中标准的值
- 本次的解题使用的是tid2013数据集，如果从Github上下载较慢，可以通过链接下载，数据集下载：[tid2013](https://www.ponomarenko.info/tid2013.htm)

## Note

- 由于`main.py`配置了多进程计算，请勿在交互式环境下运行（例如：Jupyter Notebook，IPython，Python Console等），否则会发生报错：
  
  ```
  _pickle.PicklingError: Can't pickle <function process_image at 0x00000130B7625BC0>: attribute lookup process_image on __main__ failed
  ```
  
  可以选择使用命令行来启动`main.py`，例如：
  
  ```bash
  python [path_to_main.py]
  ```
  
  或者是一般的Pycharm不配置Console来运行。
  
  - `main.py`的运行时间较长，时间取决于CPU的性能，本人运行时间大约为20~25min

