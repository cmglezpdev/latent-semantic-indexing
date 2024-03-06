# Information Retrieveral Project

### Authors:

- Alex Sanchez Saez
- Carlos Manuel Gonzales Penna

## Latent Semantic Information Model

Latent Semantic Indexing (LSI) is a sophisticated information retrieval technique that involves the analysis of relationships and patterns within a vast corpus of text. This method employs mathematical and statistical models to identify the underlying meaning and semantic structure of words, allowing for a more nuanced understanding of content.

LSI goes beyond traditional keyword-based approaches by considering the contextual relationships between words and phrases. It operates on the premise that words with similar meanings will often appear in similar contexts. Through this process, LSI creates a semantic space where words and documents are represented as vectors, facilitating a more accurate representation of content and enhancing the efficiency of information retrieval.

With this model we got from a matrix C that represents the relevance for the terms in our corpus to a matrix $C_k$ that is called a low range aproximation to make this we use SVD transformation

## How SVD decomposition works

SVD decomposition breaks the matrix in three matixes `U, Σ, and V^T`.The Σ matrix contains singular values, which represent the importance of different dimensions in the original data. By retaining only the top-k singular values and their corresponding columns in U and V^T, we perform dimensionality reduction. The U matrix contains the left singular vectors. In the context of LSI, the rows of U correspond to terms (words) in the original term-document matrix.
The columns of U represent the relationships between terms in the reduced latent semantic space. Each column can be considered a "semantic concept" that captures the underlying relationships between terms. The V^T matrix (transpose of V) contains the right singular vectors. In the context of LSI, the columns of V^T correspond to documents in the original term-document matrix.
Similar to the U matrix, the columns of V^T represent relationships between documents in the reduced latent semantic space. Each column can be seen as a "semantic concept" that captures the underlying relationships between documents

## Expanding the query

We Expand the query for a better performance and better results , in this case we use `Latent Drichlet Allocation ` for this task This method , Drichlet Allocation Method separate the terms in the corpus by topics , this uses machine learning and probabilistic inference techniques to understand the semantic relationship between the tokens in the corpus and separate then by this topics , when processing the query we use this pre-trained model to extranc the more relevant topics the query contain that are in the document , and if the topic has a certain level of relation with the query we append the terms that the topic contains to the query

## Considerations taken by us in this solution

We truncate the result SVD representation result (U, Σ, and V^T ) to take the first 500 dimensions only , we use this value because is the lowest value we test that give us good results , in the Latent Drichlet Allocation we limit the number of topics to take to the maximun of 10 , we do this in order of not adding noise to the query , and only take the terms that has more than 0.5 of probability of been within the query topics

## The Flow of the information retrieveral process

- In a First step we load our corpus , in this case we use `cranfield` , using `spacy` we process the documents and remove the stop words , noise and take only the lemma of our tokens

- Late we Build the vocabullary of our corpus and calculate the `Bag Of Words` of each document in out corpus

- With all this done , we build the term-document matrix and make SVD decomposition which returns U S and VT matrices that represent our corpus , later we truncate those matrices to get k relevant dimensions of the semantic space leaving us with U_reduces , S_reduced , Vt_reduced

- with all this set and done , we are ready to recieve the users query , when a user inputs a query in this system , the first step is to expand that query , using LDA with the pretrained model , then we process the query just like we make with the documents , and to transform the query vector to the semantic space we multiplicate it by U_reduced^T that leave us the query representation in the semantic space

- then to make the rank we go over the V_reduced matrix which is Vt_reduced^T , that matrix has the representation of the documents in the semantic space , we use cosine distance to verify the relation of the document with the query , then we take the 3 more relevant documents and send them to the ui for user to consume this documents

## Latent Semantic Indexing (LSI) Retrieval

Latent Semantic Indexing (LSI) involves Singular Value Decomposition (SVD) of the term-document matrix to capture semantic relationships. Given a term-document matrix A with dimensions m x n:

\[ A = U $\Sigma$ $V^T$ \]

where:

- \( U \) is a matrix of left singular vectors,
- \( $\Sigma$ \) is a diagonal matrix of singular values,
- \( $V^T$ \) is the transpose of a matrix of right singular vectors.

The reduced matrices \( $U_k$ \), \( $\Sigma_k$ \), and \( $V_k^T$ \) (using the top-k singular values) form the LSI representation:

\[ $A_k$ = $U_k$ $\Sigma_k$ $V_k^T$ \]

### Cosine Similarity

Cosine Similarity measures the cosine of the angle between two vectors and is often used in LSI for document similarity. Given two vectors \( A \) and \( B \):

\[ \text{Cosine Similarity}(A, B) = \frac{A $\cdot$ B}{\|A\| \|B\|} \]

where:

- \( A \cdot B \) is the dot product of vectors \( A \) and \( B \),
- \( \|A\| \) and \( \|B\| \) are the Euclidean norms of vectors \( A \) and \( B \) respectively

### Contrains Founded

In order to improve our project we could add retroalimentation to improve the quality of the retrieveral , and improve the UI , we could test to find a number of topics/singular-values that improves the performance and quality of this process , add a crawler to retrieve updated information in each use


### How to run this project 
- you can use the script to run the project by typing
  ``` bash
  ./Startup.sh
  ```
- Or see the metrics 
  ```bash
  ./Startup.sh metrics
  ```
