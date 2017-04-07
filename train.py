# training model

import text_loader
minibatch_loader = text_loader.MinibatchLoader()


minibatch_loader.load_text(([0.7, 0.15, 0.15]))
print("done")
