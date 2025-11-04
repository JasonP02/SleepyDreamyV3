# Ongoing
This marks what I have done, and what is to be done

## Current


## Next
Add symlog transform to vector inputs


## Future
* Use 99% softmax, 1% uniform for the encoder, predictor, actor
* Implement twohot 
* Implement free bits
* Implement critic learning
* Implement Actor learning
* Implement symlog and symexp for loss
* Ensure that weight initalizations are consistent with the paper.


## Completed
* Straight through gradients for passing sampled z into f(h)
* Writing the GRU block-diagonal recurrent weights
* Add tests for the GRU
* Loaded the enviornment
* Added image and vector observations
* Created encoder pipline:
* * Takes in image, vector. 
* * Runs CNN
* * Runs MLP
* * Concatenates output to produce distribution "z"
* Created the GRU architecture
* Optimize batch size etc. per https://web.eecs.umich.edu/~qstout/pap/NeskyNNBlockDiag2018.pdf

