# Ongoing
This marks what I have done, and what is to be done

## Current
Investigating how world model training will go.
* Can we train as we run the sim, or will we be sim bottlenecked?
* Should we train a policy on the env before doing the dreamer?
Implementing the loss functions for world model training
Gathering training data (obs, action) pairs

To train the world model & actor critic we need rollouts
Thus, the main script will check if we have rollouts
If there are no rollouts, we will get some
Then it will train the world model
Then we will train the actor critic inside the w/m

To me, it seems like a bad idea to couple these things too strongly. If the a/c has a problem in training, we then would need to train the w/m again. 

To get around this, we could use checkpointing. A more manual method would be just to save the model .pt file and bypass training if it exists (with the optional flag of re-training)


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

