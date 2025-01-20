

# Language Generation Using RNNs and Transformers

## Project Overview

This project aims to develop advanced language models that predict the next word in a sequence based on the context. Using a variety of deep learning techniques such as **Recurrent Neural Networks (RNNs)**, **LSTMs**, **GRUs**, and **Transformers**, we’ve built models capable of generating coherent text. We experimented with **BERT embeddings**, **Word2Vec embeddings**, and **GloVe embeddings** for word representation to improve text generation.

---

## Models Overview

- **Model 1:** BERT-RNN-based model with **BERT embeddings** (768 dimensions) used alongside stacked **RNN layers** for sequential text generation.
- **Model 2:** A series of RNN-based models (**SimpleRNN**, **LSTM**, **GRU**, and **BiLSTM**) with **pre-trained word embeddings** for next-word prediction.
- **Model 3:** Transformer-based model using **GloVe embeddings** to predict the next word based on context and incorporating **MultiHeadAttention** and **Feedforward Networks**.
- **Model 4:** Another **Transformer-based model** using **Word2Vec embeddings** with **MultiHeadAttention** and **Feedforward Networks** for better sequence modeling.

---

## Model Parameters

Here’s a summary of key model parameters for each architecture:

| **PARAMETERS**                 | **MODEL 1**          | **MODEL 2**          | **MODEL 3**          | **MODEL 4**          |
|---------------------------------|----------------------|----------------------|----------------------|----------------------|
| **Word2Vec Embedding Vector Size** | NAN                  | NAN                  | 50                   | 75                   |
| **Word2Vec Window Size**         | NAN                  | NAN                  | 2                    | 5                    |
| **Number of Encoder Layers**    | NAN                  | NAN                  | 4                    | 4                    |
| **MultiHeadAttention Heads**    | NAN                  | NAN                  | 8                    | 8 (2 × 4)            |
| **Feedforward Network Neurons** | NAN                  | NAN                  | 128                  | 128                  |
| **Activation Function**         | softmax             | ReLU, softmax        | ReLU                 | ReLU, Sigmoid        |
| **Learning Rate Initial**       | 0.001                | 0.001                | 0.05                 | 0.001                |
| **Learning Rate Decay Rate**    | NAN                  | NAN                  | 0.85                 | 0.85                 |
| **Learning Rate Decay Steps**   | 0.001                | NAN                  | 20000                | 20000                |
| **Dropout Rate**                | 0.3                  | 0.3                  | 0.1                  | 0.1                  |
| **Batch Size**                  | 64                   | 64                   | 33                   | 32                   |
| **Epochs**                      | 50                   | 30                   | 15                   | 15                   |
| **Pre-trained BERT Embedding**  | 768                  | NAN                  | NAN                  | NAN                  |

---

## Methodology

### 1. Data Preprocessing and Tokenization

To make sure the models could effectively learn from the text data, we tokenized and padded all sequences to a consistent length. We also aligned the target sequences by shifting the input tokens by one timestep for next-word prediction. For tokenization, we used **BERT tokenizer** for Models 1 and **Keras tokenizer** for the other models.

### 2. Embedding Integration

For **Model 1** and **Model 3**, we utilized **BERT embeddings** (768-dimensional) for better context extraction. For **Model 2** and **Model 4**, we employed **Word2Vec** and **GloVe embeddings** to capture the semantic meaning of words.

### 3. Model Architecture

- **Model 1 (BERT-RNN):** 
  - This model combines **BERT embeddings** with stacked **SimpleRNN layers** for sequential text generation. It uses a **softmax** activation function for predicting the next word.

- **Model 2 (RNN Variants):**
  - In this model, we experimented with **SimpleRNN**, **LSTM**, **GRU**, and **BiLSTM** architectures to determine the best method for next-word prediction. The **LSTM** variant showed the best results.

- **Models 3 and 4 (Transformer-based):**
  - These models use the **MultiHeadAttention** mechanism, allowing them to focus on different parts of the sequence simultaneously. **Model 4** uses **Word2Vec embeddings**, while **Model 3** uses **GloVe embeddings**.

### 4. Hyperparameter Tuning

The training process involved optimizing several hyperparameters. We used a **learning rate of 0.001**, **batch size of 64**, and implemented **EarlyStopping** to avoid overfitting.

### 5. Training and Evaluation

We monitored the model’s performance using **BLEU** and **ROUGE** scores. While the initial results were subpar, adjusting parameters and model configurations over time led to improvements.

---

## Challenges Faced & Decisions Made

1. **Data Preprocessing:** 
   - Tokenization and padding were essential but challenging, as sentence structures varied.
   - **Decision:** We used **BERT tokenizer** for a better understanding of context and padded sequences to a uniform length.

2. **Embedding Efficiency:** 
   - **BERT** embeddings were large and computationally demanding.
   - **Decision:** We reduced the batch size and utilized **pre-trained embeddings** to improve model efficiency.

3. **Model Complexity & Overfitting:** 
   - Complex architectures like BERT and LSTM posed risks of overfitting.
   - **Decision:** We employed **Dropout layers** and used **EarlyStopping** to prevent overfitting.

4. **Evaluation Metrics:** 
   - **BLEU** and **ROUGE** scores were low at first, showing poor text generation.
   - **Decision:** We implemented **temperature-based sampling** and tuned model parameters to improve text generation.

---

## Results

- **Model 1 (BERT-RNN):** 
  - The BLEU and ROUGE scores were initially low, but the use of **BERT embeddings** was still useful for capturing context.
  
- **Model 2 (RNN Variants):** 
  - Among **SimpleRNN**, **LSTM**, **GRU**, and **BiLSTM**, **LSTM** outperformed the rest in next-word prediction tasks.

- **Model 3 (Transformer with GloVe):**
  - Gave better results than the RNN-based models but still had room for improvement.

- **Model 4 (Transformer with Word2Vec):**
  - Achieved the best results, with a **BLEU-1 score of 0.9082** and **ROUGE-1 score of 0.4250**.

---


## Conclusion

This project demonstrates the power of combining different architectures and embeddings for language generation tasks. Despite the challenges faced, the results show that **Transformer-based models** and **pre-trained embeddings** can significantly improve text generation. There’s still room for improvement, especially in fine-tuning and model optimization.

---

## Requirements

- **Python** 3.x
- **TensorFlow** 2.x
- **Keras**
- **NumPy**
- **NLTK**
- **Huggingface Transformers** (for BERT)
- **GloVe** or **Word2Vec embeddings**

---

![InfoGraphic_LanguageGeneration1-transformed](https://github.com/user-attachments/assets/cec7af32-65c4-45b8-81ac-568f0b0018ff)






![InfoGraphic_LanguageGeneration2-transformed](https://github.com/user-attachments/assets/eea3abdb-563e-4f03-907e-9a0ea1e25a51)






