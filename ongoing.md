This marks what I have done, and what is to be done

# Ongoing

## Next
Try overfitting world model on single batch



## Future
* Use 99% softmax, 1% uniform for the encoder, predictor, actor
* Implement free bits
* Implement critic learning
* Implement Actor learning
* Ensure that weight initalizations are consistent with the paper.
* Check the use of KL divergence and distributions in general


## Completed
- Initial training loop
- Twohot encoding
Implementing loss function for decoder & reward predictor
- The image outputs a sigmoid which means we use BCE
- The vector outputs a prediction of the action distribution  

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

