# conda create -n opensloth -f environment.yaml -y

# conda %                                                                                                                            
# conda create --name unsloth_env \
#     python=3.11 \
#     pytorch-cuda=12.1 \
#     pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
#     -y
# conda activate unsloth_env

# pip install uv poetry
# uv pip install torch transformers -U
# # uv pip install 
# pip install unsloth --no-deps
# uv pip install rich
# uv pip install -e ./

curl -sSf https://astral.sh/uv/install.sh | sh
# source uv 
uv venv --python 3.11
uv pip install pip poetry
uv pip install unsloth
uv pip install -e ./

pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
