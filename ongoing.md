This marks what I have done, and what is to be done

# Ongoing

## Next
Implementing the actor critic network
I have the networks, I just need the training loop

Actions in the environment are taken from the actor network
The actor maximizes the return
The critic approximates the *distribution of returns* for each state

World model and actor generate states, actions, rewards, continue flags



## Future
* Use 99% softmax, 1% uniform for the encoder, predictor, actor
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

