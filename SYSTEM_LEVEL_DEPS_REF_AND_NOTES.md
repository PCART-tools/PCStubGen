# System-Level Dependency Reference and Notes

Refer to each project's official documentation first.

## SciPy

[Building from source](https://docs.scipy.org/doc/scipy/building/index.html)

```bash
sudo apt install gfortran libopenblas-dev liblapack-dev pkg-config
```

## Pillow

[Building from source](https://pillow.readthedocs.io/en/stable/installation/building-from-source.html)

```bash
sudo apt install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev
```

## NumPy

[Building from source](https://numpy.org/devdocs/building/)

```bash
sudo apt install gfortran libopenblas-dev liblapack-dev pkg-config
```

## psycopg2

[Build prerequisites](https://www.psycopg.org/docs/install.html#build-prerequisites)

```bash
sudo apt-get install libpq-dev
```

## UltraJSON

[Build options](https://github.com/ultrajson/ultrajson#build-options)

```bash
# No system-level dependencies are required, but debug symbols must not be stripped.
export UJSON_BUILD_NO_STRIP=1
```

## PyTorch

[Build PyTorch](https://github.com/pytorch/pytorch/wiki/Build-PyTorch)

At least 16 GB of memory is recommended.

```bash
# PCStubGen builds target with Clang/LLVM, so this dependency is required.
sudo apt install libomp-dev

# Linking may fail if this variable is not set.
export BUILD_TEST=0
```
