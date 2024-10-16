
```bash
conda create -n monolss python=3.9 pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.3 -c pytorch -c nvidia
```

```bash
conda activate monolss
```

```bash
pip3 install -r requirement.txt
```


```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```

usefull if something doesn't work:

```bash
conda info --envs
```

```bash
conda remove -n monolss --all
```
