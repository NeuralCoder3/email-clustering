# E-Mail Clustering

We use k-means clustering to cluster emails into groups. The emails are clustered based on the similarity of the words in the emails. 

1. Preprocess the emails 
    - Remove unnecessary meta information (all except from, to, subject)
    - Render the HTML as plain text
1. Tokenize the text
1. Build a bag of words vector (~ which words how often)
1. Embed the vector in a vector space
1. Perform k-means clustering
1. Visualize the results as a word cloud
1. Copy files into respective folders
