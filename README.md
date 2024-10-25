# TRUTH: Toward resolving Robot vision Understanding Task Hallucinations
This is the official code for TRUTH: Toward resolving Robot vision Understanding Task Hallucinations.

***Note***: The code was initially for resolving knowledge conflicts in Vision-Language Models with contrastive decoding, but will be revised for resolving robot vision complex scene understanding hallucination.

## Install

1. **Clone this repository**

   ```bash
   git clone https://github.com/Zhouxunzhe/TRUTH.git
   cd TRUTH
   ```

2. **Preparing conda env**

   Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:

   ```bash
   conda create -n truth python=3.9 -y
   conda activate truth
   ```

3. **conda install truth**

   ```bash
   pip install --upgrade pip
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   pip install -e .
   ```

4. **Install additional packages**

   ```bash
   pip install flash-attn --no-build-isolation
   ```

5. Upgrade to the latest code base

   ```bash
   git pull
   pip install -e .
   ```

## Run

For different models

```
cd [your model name]
# for example, 'cd tiny_llava' for model Tiny LLaVA
```

Run demo.py for quick inference

```
python demo.py
```

Run main.py for contrastive decoding

```
python main.py
```

Set alpha and contrast_layer_id for contrastive decoding

```
alpha = 0.5
contrast_layer_id = 21
```

