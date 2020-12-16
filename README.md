# Multi-Hop Question Answering

## Instructions
- To view answer prediction results on test data with trained weights or baseline model, use the Colab notebooks from the resources below.
- To train a fresh model with the following configurations, use utils/model_trainer.py
  - **Configurations**\
      Tokenizer Max Length = 1024\
      Epochs = 2\
      Learning Rate = 0.00005\
      Architecture Name = allenai/longformer-base-4096\
      Save Name for Weights = neew_weights\
  - **Command**\
  ```python model_trainer.py 1024 8 2 0.00005 allenai/longformer-base-4096/ new_weights```\

- The data pre-processor script and the data splitter script can be found in utils/model_trainer

## Resources

### Data set
https://drive.google.com/file/d/1pFJ0NAvMSn7C-vI-hzeSsCG7ppJGa8EV/view?usp=sharing

### Tokenizer
https://drive.google.com/file/d/1Ra6HNBnP8bGutLi7076Vk0kizznjOL-X/view?usp=sharing

### Model
https://drive.google.com/file/d/1DMWe7bLI0FZ6Qd5tUyIYBCWMy6uLrHy5/view?usp=sharing

### Interactive Notebooks (NYU Account Only)

#### Baseline Model Inference Notebook
https://colab.research.google.com/drive/1Yn7ARYrp3JGKNeBrTnuL2XMMvVkpQwto?authuser=1
#### Trained Model Inference Notebook
https://colab.research.google.com/drive/10B71qnh9oAkeWJ7x71dTOABJh92KxHGs?authuser=1
