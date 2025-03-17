set +e  # 关闭严格模式，允许命令失败后继续执行

export CONDA_ENV_NAME=hologest_env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.8 || true

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME || true

which python
which pip

sudo apt-get install libturbojpeg || true

pip install --upgrade torch torchvision torchaudio || true
pip install lmdb -i https://mirrors.tencent.com/pypi/simple/ || true
pip install timm -i https://mirrors.tencent.com/pypi/simple/ || true
pip install wandb -i https://mirrors.tencent.com/pypi/simple/ || true
pip install IPython -i https://mirrors.tencent.com/pypi/simple/ || true
pip install librosa -i https://mirrors.tencent.com/pypi/simple/ || true
pip install pyarrow==10.0.0 -i https://mirrors.tencent.com/pypi/simple/ || true
pip install easydict -i https://mirrors.tencent.com/pypi/simple/ || true
pip install configargparse -i https://mirrors.tencent.com/pypi/simple/ || true
pip install einops -i https://mirrors.tencent.com/pypi/simple/ || true
pip install omegaconf -i https://mirrors.tencent.com/pypi/simple/ || true
pip install transformers==4.30.2 -i https://mirrors.tencent.com/pypi/simple/ || true
pip install ftfy -i https://mirrors.tencent.com/pypi/simple/ || true
pip install regex -i https://mirrors.tencent.com/pypi/simple/ || true
pip install blobfile -i https://mirrors.tencent.com/pypi/simple/ || true
pip install h5py -i https://mirrors.tencent.com/pypi/simple/ || true
pip install pandas -i https://mirrors.tencent.com/pypi/simple/ || true
pip install loguru -i https://mirrors.tencent.com/pypi/simple/ || true

pip install pytorch3d -i https://mirrors.tencent.com/pypi/simple/ || true
pip install transformers -i https://mirrors.tencent.com/pypi/simple/ || true

pip install "chumpy" -U --index-url https://mirrors.tencent.com/pypi/simple/ || true
pip install "p-tqdm"  --index-url https://mirrors.tencent.com/pypi/simple/ || true
pip install "accelerate" -U --index-url https://mirrors.tencent.com/pypi/simple/ || true
pip install "smplx" -U --index-url https://mirrors.tencent.com/pypi/simple/ || true
pip install "clip" -U --index-url https://mirrors.tencent.com/pypi/simple/ || true
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" || true


cd ./hologest/CLIP/
python setup.py install || true
pip install -r requirements.txt || true


cd ./holgest/MotionPrior
pip install -r requirements.txt || true