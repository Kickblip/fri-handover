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


To run model that predicts 5 frames into the future for the receiving hand and can generate a video visualization. Run:
# Train
```
python -m model.train
```
#Inference with video
```
python -m model.infer 1_w_b --video
```
The video will be saved to dataset/model_output/videos/<stem>_predicted_future.mp4 showing the predicted future frames of the receiving hand.
