atest_pic                                     测试图片
atest_video                                   测试视频

 
                       train/valid/test图片数量
helmet_dataset         2043/195/97            头盔数据集
helmet_vest_dataset    237/17/17              头盔背心数据集1
helmet_vest_dataset2   2111/510/171           头盔背心数据集2
fire_smoke_dataset     3320/684/300
monitor_worker         2408/806/131
-test                                         测试集
-train                                        训练集 
-valid                                        验证集
-data.yaml                                    训练时的配置文件
-train.py                                     训练数据集



venv                                          python环境

helmet_model                                  头盔数据集训练结果
helmet_vest_model                             头盔背心数据集1训练结果
helmet_vest_model2                            头盔背心数据集2训练结果

test_model                                    测试模型

yolo26n                                       训练数据集所需基础模型 
yolo8n                                        训练数据集所需基础模型
-------------------------------------------------------------------------------------------------------------------
使用说明:

测试结果:test_model.py运行程序，14行开始选择一个训练好的模型加载,以pt为后缀

训练模型:选择一个数据集，运行train.py，在项目目录上级目录下找到\runs\detect\[project]\[name]\weights\best.pt即为训练好的模型
其中project,name参数在train.py18行



