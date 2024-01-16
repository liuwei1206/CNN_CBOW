# CNN_CBOW
This is an implementation of CBOW using Pytorch.

This implementation is a little different from the trainditional one. In the original version of CBOW, context was defined as the average embeddings of words around the central token. Here, I use a **convolution moudle** over around words to get the representation of context.

Here are some examples to show the quality of our trained embeddings (Sorry, you need to know Chinese :)).

![image](https://github.com/liuwei1206/CNN_CBOW/blob/master/images/result.png)



#########################   Description   ###########################

main.py

export_embed.py: export embeddings from the model

data: the data files

cn_cove: the main code of the model


