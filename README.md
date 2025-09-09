```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd manopth
pip install .
```


pip installing chumpy directly causes import errors for SMPL models.
installing from github for now since this isnt reflected on pypi:
https://github.com/mattloper/chumpy/pull/48