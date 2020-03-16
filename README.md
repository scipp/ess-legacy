# scipp-ess

This repository contains [ESS](https://europeanspallationsource.se/)-specific code building on [scipp](https://github.com/scipp/scipp).

**Own-cloud Data**

The reduce.py script will attempt to discover the root script directory via a config.py (non-versioned). Use make.py to set the local script root directory i.e:

```python
python make.py /home/usr/owncloud/RAL_Mantid_September/data_GP2/processing/
```
Also see:
```
python make.py --help
```
