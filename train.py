# training model

import text_loader
minibatch_loader = text_loader.MinibatchLoader()


minibatch_loader.load_text(([0.7, 0.15, 0.15]), 2)
print("done")
print(minibatch_loader.batch_pointers)
print(minibatch_loader.data)
#print(minibatch_loader.next_batch(0))
#print(minibatch_loader.next_batch(0))