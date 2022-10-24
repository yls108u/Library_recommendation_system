# Using Course selection records to solve cold start problem of the recommender system for library book 


**Due to privacy issue, we cannot public the course selection data and book lending data.**


## ```preprocessing.ipynb```: 
It's for data preprocessing, including data mining and cleaning.

## ```exploration.ipynb```: 

To give the evidences to show that __course selection records__ and __book lending records__ have some relationship.

Note : The dense matrices ```usercoursedense``` and ```cate3_userdense``` are constructed during exploration part.

### The report is at:
https://docs.google.com/presentation/d/10_HgqXmyc8QRP4uh6VgHjHQczXp69FkZ/edit?usp=sharing&ouid=104583669107890199313&rtpof=true&sd=true

## ``` recommendation.ipynb```:

Doing recommendation.
The algorithm is navie __Collaborative Filter__ (CF).

### The report is at:
https://docs.google.com/presentation/d/1iA2pFEpgqHyy1AKEl-TOZRX6D9lZ4Kss/edit?usp=sharing&ouid=104583669107890199313&rtpof=true&sd=true


## ``` evaluation.py```:

caculating the metrices $\text{precision},\text{recall},\text{F1-score}$ and plotting ROC, PR curve.




## The visualization can be found at the director: ./result 