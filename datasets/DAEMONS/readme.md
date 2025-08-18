# Potsdam Dataset for Eye Movement On Natural Scenes (Potsdam DAEMONS) 
## Eye Tracking Data

The folder contains 2 csv files:
    - training data
    - validation data
the test set is being withheld for use in a benchmark.

### Rows
Each row in the csv file represents one saccade and the subsequent fixation. In other words, each row is a saccade and the fixation that it leads to.
This means that for the first line in each trial it is very likely that the saccade information is missing (NA) because the trial starts with a fixation, and that fixation has no preceding saccade.
Likewise, the last event in each sequence may be incomplete, if the recording stopped while the participant was performing a saccade.

### Columns
Here, we provide a description of each column in the data set.

- **Start**: start time of the saccade relative to the trial start
- **End**: end time of the saccade relative to the trial start
- **sacdur**: duration of the saccade
- **delay**: delay between eyes
- **PeakVel**: Peak velocity of the saccade
- **sacLen**: saccade length, i.e. fixation endpoint to  fixation endpoint
- **Ampl**: saccade amplitude, i.e. full extent of the saccade including overshoot
- **VP**: Subject ID
- **trial**: Trial number
- **Img**: Image Name
- **nth**: number of the event in the sequence (trial).
- **x**: x coordinate in degrees of visual angle. The origin is bottom left.
- **y**: y coordinate in degrees of visual angle. The origin is bottom left.
- **fixdur**: Fixation duration in ms.
- **sticky**:
- **blinkSac**: A blink occurred during this saccade.
- **blinkFix**: A blink occurred during this fixation.
- **first_t**: timestamp of the trial start in raw data coordinates
- **forced_fix**: 1 if this event occurred during the forced fixation time.
- **VT**: membership- either "test", "train", or "val".

### trial start

Each trial began with a fixation marker. The stimulus image appeared underneath the marker and participants were asked to keep fixating the marker until it disappeared. If participants moved their eyes during this interval a mask was presented and the fixation check was restarted. In this case we only repost data from the last, successful fixation check.
