# APAI3010_group5_mast3r
This is the repo for APAI3010 project of group 25, cite and repair [MAST3R](https://github.com/naver/mast3r).

## Here is how you can correctly run the process of this project:
### step 1: clone this repo and create a conda environment:
```bash
git clone https://github.com/Mommrie/APAI3010_group5_mast3r.git
cd APAI3010_group5_mast3r
```
remind: there are some changed files in our project, different from original files in mast3r repo, so following commands may appear errors if you git clone from mast3r repo.
```bash
conda create -n mast3r python=3.9 cmake=3.14.0
conda activate mast3r
```

```bash
conda install pytorch torchvision pytorch-cuda=12.1 faiss-gpu -c pytorch -c nvidia -c conda-forge
conda install habitat-sim headless -c conda-forge -c aihabitat
conda install scipy scikit-learn einops opencv pillow-heif pyyaml -c conda-forge
conda install -c conda-forge quaternion
pip install trimesh pyrender 'pyglet<2' roma poselib imageio-ffmpeg
pip install pycolmap kapture kapture-localization
pip install 'huggingface-hub[torch]>=0.22' gradio
conda install matplotlib tensorboard notebook ipykernel ipywidgets widgetsnbextension tqdm -c conda-forge
pip install cython
```

Sometimes there will be an error after this. Enter the command according to the prompt: 
```bash
apt-get update
apt-get install -y libegl1 libgl1-mesa-glx
```

Then you can use ```pip check``` to determine if there are any conflicts. Moreover, try:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('GPU device:', torch.cuda.get_device_name(0))"
python -c "import habitat_sim; print('Habitat Sim imported successfully. Version:', habitat_sim.__version__)"
```

### step 2: Fill in the required files according to the mast3r source code base (see: https://github.com/naver/mast3r?tab=readme-ov-file#installation):
```bash
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
python3 setup.py build_ext --inplace
cd ..
```

```bash
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

### step 3: repair the code and lib to successfully run our model:
We might do some change in python lib ```gradio-cilent```, corresponding path is under the created virtual environment folder. 
(for example: ```miniconda3/envs/mast3r/lib/python3.9/site-packages/gradio_client/utils.py```)

This is because: <br>
Bool ```True``` or ```false``` is not a legal JSON schema object, but it does exist in some schemas when running this model (we have no idea how it appears); <br>
Gradio does not rely on this schema=true to provide structured fields, so returning "any" is a safe degradation process;

The first is about 860 lines of this file, original code: 
```python
def get_type(schema):
    if "const" in schema:
        return "const"
    if "enum" in schema:
        return "enum" ...
```
add a ```if``` condition, change to:
```python
def get_type(schema):
    if not isinstance(schema, dict):
        return "Unknown"
    if "const" in schema:
        return "const"
    if "enum" in schema:
        return "enum" ...
```

The second one is about line 899, original code:
```python
def _json_schema_to_python_type(schema: Any, defs) -> str:
    """Convert the json schema into a python type hint"""
    if schema == {}:
        return "Any"
    type_ = get_type(schema)
    if type_ == {}: ...
```
add a ```if``` condition, change to:
```python
def _json_schema_to_python_type(schema: Any, defs) -> str:
    """Convert the json schema into a python type hint"""
    if isinstance(schema, bool):
        return "Any"
    if schema == {}:
        return "Any"
    type_ = get_type(schema)
    if type_ == {}: ...
```
Remind to save these changes.

### other error:
Generally speaking, running the command will automatically download a file named ```frpc_linux_amd64_v0.2```, but sometimes if the automatic download fails, you need to manually download it and put it into the corresponding ```gradio``` library <br>
(for example: ```miniconda3/envs/mast3r/lib/python3.9/site-packages/gradio/frpc_linux_amd64_v0.2```).

### finally, we can run this model:
use command if you use model provided by mast3r like us:
```bash
python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --share
```

After compilation, the following dialog will be generated in bash (for example):
```bash
[2025-04-19 14:36:34] <All keys matched successfully>
[2025-04-19 14:36:34] Outputing stuff in /tmp/tmpnm6w3f9g_mast3r_gradio_demo/b7a042df61818cd21f2f14a48aff5f86
[2025-04-19 14:36:34] Using Gradio version: 4.44.1
[2025-04-19 14:36:35] Running on local URL:  http://127.0.0.1:7860
[2025-04-19 14:36:36] Running on public URL: https://2d2cd47e0dbebe6641.gradio.live
[2025-04-19 14:36:36] 
This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```

These are the links of gradio interactive page: ```http://127.0.0.1:7860 # for local web``` and ```https://xxx.gradio.live # for public web```.
open either of them you can see website as:

![](https://github.com/Mommrie/APAI3010_group5_mast3r/blob/main/demo_image.jpeg)
