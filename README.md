# demo_for_nlp_interface
中文SimpleTOD結合R-Drop並實作demo介面

**此為Braslab實驗室使用**


#### 預先安裝的套件及前導知識
 1. torch
 2. gradio
```
pip install gradio
```
3. transformer
安裝連結[huggingface transformers](https://huggingface.co/docs/transformers/installation#install-from-source)
---
1. 了解如何使用transformers的模型載入跟使用
詳情可參考[這邊](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer)
2. 閱讀SimpleTOD論文[SimpleTOD](https://arxiv.org/pdf/2005.00796.pdf)

#### demo用程式
1. demo_with_interface
2. 模型權重則要自行訓練

#### 其他補充
1. demo.py為英文版的demo(無介面)
2. demo_zhtw_mod.py為測試修改中文用(無介面)
3. 模型權重訓練(待補...)


