# Ongoing
This marks what I have done, and what is to be done

## Current
Writing the GRU
Seems like I need to revise GRU inputs, and hidden state dimensions
Also unclear on what the 

## Next
Implement the dynamics predictor (finalizing RSSM)

## Future
* Add tests for the GRU
* Use 99% softmax, 1% uniform for the encoder, predictor, actor
* Implement twohot 
* Implement free bits
* Implement critic learning
* Implement Actor learning
* Implement symlog and symexp for loss
* Ensure that weight initalizations are consistent with the paper.


## Completed
* Loaded the enviornment
* Added image and vector observations
* Created encoder pipline:
* * Takes in image, vector. 
* * Runs CNN
* * Runs MLP
* * Concatenates output to produce distribution "z"
* Created the GRU architecture
