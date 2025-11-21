# News Article Categorization with LSTM

So I built this news classifier using LSTM networks, and honestly it works pretty well. The idea was to take news articles and automatically figure out which category they belong to - World, Sports, Business, or Science/Tech.

## What's this about?

This project uses the AG News dataset to train a bidirectional LSTM model that can categorize news articles. I spent some time experimenting with different architectures and this one gave me the best results.

The model looks at the title and description of a news article and predicts which of the 4 categories it falls into:
- World News
- Sports News  
- Business News
- Science & Technology News

## The Dataset

I'm using the AG News Classification Dataset which has 120,000 training samples and 7,600 test samples. Each article has a title and description, which I concatenated together for training. The dataset is pretty balanced across all 4 categories which made training easier.

## How it works

The approach is fairly straightforward:

1. **Text Preprocessing**: I tokenized the text and limited the vocabulary to 10,000 words (seemed like a good middle ground). Used padding to make all sequences the same length.

2. **Model Architecture**: 
   - Embedding layer (32 dimensions)
   - Two stacked Bidirectional LSTM layers (128 and 64 units)
   - GlobalMaxPooling to extract features
   - Dense layers with dropout for regularization
   - Softmax output for the 4 categories

3. **Training**: Used sparse categorical crossentropy since I didn't one-hot encode the labels. Added early stopping and model checkpointing to save the best weights based on validation accuracy.

## Running the code

The whole thing is in a Jupyter notebook (`ag-news-classification-lstm.ipynb`). You'll need:

```
pandas
numpy
matplotlib
tensorflow/keras
scikit-learn
wandb (for experiment tracking)
```

Just run the notebook cells in order. Note that you'll need the AG News dataset - the notebook expects it in `../input/ag-news-classification-dataset/`.

## Results

The model achieves pretty decent accuracy on the test set. I tracked all my experiments using Weights & Biases so I could compare different hyperparameters.

You can test it out with your own news headlines - there's a `modelDemo()` function at the end that takes any text and predicts its category.

## Example

```python
modelDemo(['New evidence of virus risks from wildlife trade'])
# Output: Science-Technology News

modelDemo(['Trump\'s bid to end Obama-era immigration policy ruled unlawful'])
# Output: World News
```

## Things I learned

- Bidirectional LSTMs work better than unidirectional for this task
- GlobalMaxPooling after LSTMs helps reduce overfitting
- The vocab size didn't need to be huge - 10k words was enough
- Early stopping saved me a lot of training time

## What could be better

If I revisit this, I'd probably try:
- Using pretrained word embeddings (GloVe or Word2Vec)
- Experimenting with attention mechanisms
- Fine-tuning a transformer model (though that might be overkill)

Feel free to fork this and experiment! Would love to see what accuracy you can get.
