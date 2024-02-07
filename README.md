# Neural Network to Classify a Rat's Lever-Press State Based on Neural Firing Information
## Final Project for ELEC3810 (Data Science for Neural Engineering) at HKUST in Fall 2023
### Experiment Paradigm
When there is a start cue, the rat is supposed to press the lever. Then a water reward will be given to him.
### Objective
Classify the rat's lever-press state (press or rest) from his neural firing information (spike).
### Data Description
trainSpike (16 * 13145) - The rows represents different channels. The columns represent the time sequence of the experiment. Each column means 100ms, so our training experiment lasts for 1314.5s. The value of the matrix represents the spike count in that time window.

trainState (1 * 14152) - The column represents the same meaning as that of trainSpike. The value 0 means, at that time, the rat is in the rest state, 1 means the rat is in press state, and NaN means the rat'e state is not clear.

testSpike (16 * 3656) - The row represents different channel. The column represents the time sequence of the experiment. Each column means 100 ms. The value of the matrix represents the spike count in that time window. You need to decode the rat’s state for all the timestamp in testSpike (3656 states in total).
