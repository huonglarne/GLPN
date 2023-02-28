# GLPN
This repo runs Huggingface's implementation of [GLPN](https://huggingface.co/docs/transformers/model_doc/glpn).

The code is borrowed from this [tutorial](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/GLPN).

# Run with Moreh framrework
        conda create -n glpn python=3.8
        conda activate glpn
        update-moreh --force --targer 23.3.0
        pip install matplotlib

# Inference
        python glpn.py