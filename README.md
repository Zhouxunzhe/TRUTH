# TRUTH: Toward resolving Robot vision Understanding Task Hallucinations
This is the official code for TRUTH: Toward resolving Robot vision Understanding Task Hallucinations.

## Install

Clone this repository and navigate to LLaVA folder

```
git clone https://github.com/DLCV-BUAA/TinyLLaVABench.git
cd TinyLLaVABench
```

Install Package

```
conda create -n tinyllava python=3.10 -y
conda activate tinyllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Upgrade to the latest code base

```
git pull
pip install -e .

# if you see some import errors when you upgrade, please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
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

