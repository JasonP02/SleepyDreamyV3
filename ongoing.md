# Ongoing
This marks what I have done, and what is to be done

## Current
Figuring out how to pass in the encoder output (z) to the RSSM (f(h_{t-1}...))
The output of the encoder is (B,d_h,n_bins) where n_bins = d_h/16
The input of the RSSM is 

The key detail is the usage of 'straight through gradients'. This is from *Dreamer v2* under **algorithm 1**



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

