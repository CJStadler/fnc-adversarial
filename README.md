# Generating adversarial examples for the Fake News Challenge

The baseline FNC implementation is included as a submodule of this project (and
the FNC is a submodule within that). To pull it run this
```
git submodule update --init --recursive
```

The `deletions.py` script generates misclassified examples by deleting words
from examples that were correctly classified by the baseline model. To run it:
```
python3 deletions.py
```
The first run will be very slow because it needs to generate the features for
every example. These are cached so future runs should be faster (~30min).

## Results
```
Original correct agree/disagree: 180
Prediction changed count: 46
Median deletions: 5.0
```
