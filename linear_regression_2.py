from Utility.Timer import Timer
import Utility.Utility as Utility
from DataModules.CH11DataModule import CH11DataModule
from Models.LinearRegression import LinearRegression
from Models.Scratch.LinearRegressionScratch import LinearRegressionScratch
from Trainer.Trainer import Trainer
from Models.Scratch.SGD import SGD
from Models.Scratch.Adagrad import Adagrad
from Models.Scratch.RMSProp import RMSProp
from Models.Scratch.Adadelta import Adadelta
import torch
import math

from ProgressBoard import ProgressBoard
#import ProgressBoard
import matplotlib.pyplot as plt


board = ProgressBoard()
board.xlabel='epoch'

sgd_scratch_optim = lambda params: SGD(params, 0.01, momentum=0.9)
sgd_optim = lambda params: torch.optim.SGD(params, 0.01, momentum=0.9)

adagrad_scratch_optim = lambda params: Adagrad(params, lr=0.01)
adagrad_optim = lambda params: torch.optim.Adagrad(params, lr = 0.01)

rms_prop_scratch_optim = lambda params: RMSProp(params, lr=0.01, gamma=0.9)
rms_prop_optim = lambda params: torch.optim.RMSprop(params, lr=0.01, alpha=0.9)

adadelta_scratch_optim = lambda params: Adadelta(params, rho=0.9)
adadelta_optim = lambda params: torch.optim.Adadelta(params, rho=0.9)
models = [
    #(LinearRegressionScratch(num_inputs=5, optimizer=sgd_scratch_optim), "linear_scratch"),
    #(LinearRegression(optimizer=sgd_optim), "linear"),
    #(LinearRegressionScratch(num_inputs=5, optimizer=adagrad_scratch_optim), "adagrad_scratch"),
    #(LinearRegression(optimizer=adagrad_optim), "adagrad"),
    #(LinearRegressionScratch(num_inputs=5, optimizer=rms_prop_scratch_optim), "rms_prop_scratch"),
    #(LinearRegression(optimizer=rms_prop_optim), "rms_prop")
    (LinearRegressionScratch(num_inputs=5, optimizer=adadelta_scratch_optim), "adadelta_scratch"),
    (LinearRegression(optimizer=adadelta_optim), "adadelta")
]

for model,model_name in models:
    trainer = Trainer(max_epochs=10, num_gpus=1)
    ch11 = CH11DataModule(batch_size = 10)
    trainer.fit(model, ch11)
    stats = trainer.get_stat("epoch", "epoch_x", "loss")
    for (x, y) in stats:
        board.draw(x, y, label=model_name)


plt.show()

#board = ProgressBoard()
#board.xlabel='epoch'
#tests = [{"lr":0.02, "momentum" : 0.5},
    #{"lr":0.01, "momentum": 0.9},
    #{"lr":0.005, "momentum": 0.9}]
#
#for test in tests:
    #ch11 = CH11DataModule(batch_size = 10)
#
    #model = LinearRegressionScratch(num_inputs=5, lr = test["lr"], momentum=test["momentum"])
    ##model = LinearRegression(lr = 0.02, momentum=0.5)
    #trainer = Trainer(max_epochs=10, num_gpus=1)
    #trainer.fit(model, ch11)
    #stats = trainer.get_stat("epoch", "epoch_x", "loss")
    #for (x, y) in stats:
        #board.draw(x, y.to(torch.device('cpu')).detach().numpy(), label=f'lr:{test["lr"]}, momentum:{test["momentum"]}')
#
    #epoch_loss = trainer.get_stat("epoch", "epoch_x", "duration")
    #print(epoch_loss)
#plt.show()

#ProgressBoard.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         #'time (sec)', 'loss', xlim=[1e-2, 10],
         #legend=['gd', 'sgd', 'batch size=100', 'batch_size=10'])
         #
         #
#board = ProgressBoard()
#board.xlabel='epoch'
#data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.3)
#model = LinearRegressionScratch(lr=0.01, num_inputs=2)
#trainer = Trainer(max_epochs=5)
#trainer.fit(model, data)
#stats = trainer.get_stat("train_batch", "epoch_x", "loss")
#for (x, y) in stats:
    #board.draw(x, y.to(torch.device('cpu')).detach().numpy(), label="asdf")
#plt.show()


#plt.show()

#def f_2d(x1, x2):
    #return 0.1 * x1 ** 2 + 2 * x2 ** 2
#
#eta = 0.4
#def gd_2d(x1, x2, s1, s2):
    #return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)
#
#eta, beta = 0.6, 0.5
#def momentum_2d(x1, x2, v1, v2):
    #v1 = beta * v1 + 0.2 * x1
    #v2 = beta * v2 + 4 * x2
    #return x1 - eta * v1, x2 - eta * v2, v1, v2
#
#eta = 0.4
#def adagrad_2d(x1, x2, s1, s2):
    #eps = 1e-6
    #g1, g2 = 0.2 * x1, 4 * x2
    #s1 += g1 ** 2
    #s2 += g2 ** 2
    #x1 -= eta / math.sqrt(s1 + eps) * g1
    #x2 -= eta / math.sqrt(s2 + eps) * g2
    #return x1, x2, s1, s2
   # 
#
#eta, gamma = 0.4, 0.9
#def rmsprop_2d(x1, x2, s1, s2):
    #g1, g2, eps = 0.2 * x1,  4 * x2, 1e-6
    #s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    #s2 = gamma * s2 + (2 - gamma) * g2 ** 2
    #x1 -= eta / math.sqrt(s1 + eps) * g1
    #x2 -= eta / math.sqrt(s2 + eps) * g2
    #return x1, x2, s1, s2
#
#ProgressBoard.show_trace_2d(f_2d, Utility.train_2d(rmsprop_2d))
#plt.show()


#betas = [0.95, 0.9, 0.6, 0]
#for beta in betas:
    #x = torch.arange(40).detach().numpy()
    #plt.plot(x, beta ** x, label = f'beta ={beta:.2f}')
    #plt.xlabel('time')
    #plt.legend()
#plt.show()
    

#plt.show()
#gd_params = {"batch_size" : 1500, "num_epochs":10, "lr" : 0.1 }
#sgd_params = {"batch_size" : 1, "num_epochs": 2, "lr" : 0.005}
#mini1_params = {"batch_size" : 100, "num_epochs" : 2, "lr" : 0.4}
#mini2_params = {"batch_size" : 10, "num_epochs" : 2, "lr" : 0.05}
#
#
#training_params = [
    #{
        #"name" : "gd",
        #"params" : gd_params
    #},
    #{
        #"name" : "sgd",
        #"params" : sgd_params
    #},
    #{
        #"name" : "batch size 100",
        #"params" : mini1_params
    #},
    #{
        #"name" : "batch size 10",
        #"params" : mini2_params
    #}
#]



#board = ProgressBoard()
#plt.gca().set_xscale('log')
#board.xlabel = 'time '
#for training_param in training_params:
    #batch_size = training_param["params"]["batch_size"]
    #lr = training_param["params"]["lr"]
    #num_epochs = training_param["params"]["num_epochs"]
   # 
    #ch11 = CH11DataModule(batch_size = batch_size, num_train = 1500)
#
#
#
#
#
    #model = LinearRegression(lr = lr)
    #trainer = Trainer(max_epochs=num_epochs, num_gpus=1)
    #trainer.fit(model, ch11)
#
    #stats = trainer.get_stat("train_batch", "rel_start_time", "loss")
    #for (x, y) in stats:
        #board.draw(x, y.to(torch.device('cpu')).detach().numpy(), label=training_param["name"])
#
#plt.show()

#ProgressBoard.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         #'time (sec)', 'loss', xlim=[1e-2, 10],
         #legend=['gd', 'sgd', 'batch size=100', 'batch_size=10'])

#timer = Timer()
#
## comp
#
#B = torch.randn(256, 256)
#C = torch.randn(256, 256)
#
#A = torch.zeros(B.shape[0], C.shape[1])
#
#timer.start()
#for i in range(A.shape[0]):
    #for j in range(A.shape[1]):
        #A[j, j] = torch.dot(B[i, :], C[:,j])
#print(timer.stop())
#
## Compute A = BC one column at a time
#timer.start()
#for j in range(A.shape[1]):
    #A[:, j] = torch.mv(B, C[:, j])
#print(timer.stop())
#
## Computer A = BC in one go
#timer.start()
#A = torch.mm(B, C)
#print(timer.stop())
#
#gigaflops = [0.03 / i for i in timer.times]
#print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      #f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
#
#timer.start()
#for j in range(0, 256, 64):
    #A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
#timer.stop()
#print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
    

#def f(x1, x2):  # Objective function
    #return x1 ** 2 + 2 * x2 ** 2
#
#def f_grad(x1, x2):  # Gradient of the objective function
    #return 2 * x1, 4 * x2
#
#def sgd(x1, x2, s1, s2, f_grad):
    #g1, g2 = f_grad(x1, x2)
    ## Simulate noisy gradient
    #g1 += torch.normal(0.0, 1, (1,)).item()
    #g2 += torch.normal(0.0, 1, (1,)).item()
    #eta_t = eta * lr()
    #return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
#
#def constant_lr():
    #return 1
#
#def exponential_lr():
    #global t
    #t += 1
    #return math.exp(-0.1*t)
#
#def polynomial_lr():
    #global t
    #t += 1
    #return (1 + 0.1 * t) ** (-0.5)
#
#eta = 0.1
##lr = constant_lr  # Constant learning rate
#t = 1
##lr = exponential_lr
#lr = polynomial_lr
#ProgressBoard.show_trace_2d(f, Utility.train_2d(sgd, steps=50, f_grad=f_grad))
#plt.show()

#def f_2d(x1, x2):  # Objective function
    #return x1 ** 2 + 2 * x2 ** 2
#
#def f_2d_grad(x1, x2):  # Gradient of the objective function
    #return (2 * x1, 4 * x2)
#
#def gd_2d(x1, x2, s1, s2, f_grad):
    #g1, g2 = f_grad(x1, x2)
    #return (x1 - eta * g1, x2 - eta * g2, 0, 0)
#
#eta = 0.1
#ProgressBoard.show_trace_2d(f_2d, Utility.train_2d(gd_2d, f_grad=f_2d_grad))
#plt.show()

#f = lambda x: 0.5 * x**2  # Convex
#g = lambda x: torch.cos(np.pi * x)  # Nonconvex
#h = lambda x: torch.exp(0.5 * x)  # Convex
#
#x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
#ProgressBoard.use_svg_display()
#_, axes = plt.subplots(1, 3, figsize=(9, 3))
#for ax, func in zip(axes, [f, g, h]):
    #ProgressBoard.plot([x, segment], [func(x), func(segment)], axes=ax)
#
#plt.show()

#x = torch.arange(-2.0, 5.0, 0.01)
#ProgressBoard.plot(x, [torch.tanh(x)], 'x', 'f(x)')
#ProgressBoard.annotate('vanishing gradient', (4, 1), (2, 0.0))
#plt.show()
#x, y = torch.meshgrid(
    #torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101)
#)
#z = x**2 - y**2
#
#ax = ProgressBoard.plt.figure().add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
#ax.plot([0], [0], [0], 'rx')
#ticks = [-1, 0, 1]
#plt.xticks(ticks)
#plt.yticks(ticks)
#ax.set_zticks(ticks)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()
#x = torch.arange(-2.0, 2.0, 0.01)
#ProgressBoard.plot(x, [x**3], 'x', 'f(x)')
#ProgressBoard.annotate('saddle point', (0, -0.2), (-0.52, -5.0))
#plt.show()
#def f(x):
    #return x * torch.cos(np.pi * x)
#x = torch.arange(-1, 2.0, 0.01)
#ProgressBoard.plot(x, [f(x), ], 'x', 'f(x)')
#ProgressBoard.annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
#ProgressBoard.annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
#plt.show() 

#def f(x):
    #return x* torch.cos(np.pi * x)
#
#def g(x):
    #return f(x) + 0.2 * torch.cos(5 * np.pi * x)
#x = torch.arange(0.5, 1.5, 0.01)
#ProgressBoard.set_figsize((4.5, 2.5))
#ProgressBoard.plot(x, [f(x), g(x)], 'x', 'risk')
#ProgressBoard.annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
#ProgressBoard.annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
#plt.show()

#img_size, patch_size = 96, 16
#num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
#emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
#model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout, lr)
#trainer = Trainer(max_epochs=10, num_gpus=1)
#data = FashionMNIST(batch_size=128, resize=(img_size, img_size))
#trainer.fit(model, data)

#X = torch.ones((2, 100, 24))
#encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
#encoder_blk.eval()
#Utility.check_shape(encoder_blk(X), X.shape)
#img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
#patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
#X = torch.zeros(batch_size, 3, img_size, img_size)
#Utility.check_shape(patch_emb(X), (batch_size, (img_size//patch_size)**2, num_hiddens))


#data = MTRusEng(batch_size = 128)
#data = MTFraEng(batch_size = 128)
#num_hiddens, num_blks, dropout = 256, 2, 0.2
#ffn_num_hiddens, num_heads = 64, 4
#encoder = TransformerEncoder(
    #len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
#
#decoder = TransformerDecoder(
    #len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
#
#model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.001)
#
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#
#engs = ['i see .', 'smile .', 'buy it .', 'wow .']
#fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

#engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
#fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
#
#
#preds,_ = model.predict_step(data.build(engs, fras), trainer.device(), data.num_steps)
#for en, fr, p in zip(engs, fras, preds):
    #translation = []
    #for token in data.tgt_vocab.to_tokens(p):
        #if token == '<eos>':
            #break
        #translation.append(token)
    #print(f'{en} => {translation}, bleu, {Utility.bleu(" ".join(translation), fr, k=2)}')
#X = torch.ones((2, 100, 24))
#
#_, dec_attention_weights = model.predict_step(
    #data.build([engs[-1]], [fras[-1]]), trainer.device(), data.num_steps, True)
#enc_attention_weights = torch.cat(model.encoder.attention_weights, 0)
#shape = (num_blks, num_heads, -1, data.num_steps)
#enc_attention_weights = enc_attention_weights.reshape(shape)
#Utility.check_shape(enc_attention_weights, (num_blks, num_heads, data.num_steps, data.num_steps))
#
#show_heatmap(
    #enc_attention_weights.cpu(), xlabel='Key positions',
    #ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    #figsize=(7, 3.5))
#
#
#dec_attention_weights_2d = [head[0].tolist()
                            #for step in dec_attention_weights
                            #for attn in step for blk in attn for head in blk]
#dec_attention_weights_filled = torch.tensor(
    #pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
#shape = (-1, 2, num_blks, num_heads, data.num_steps)
#dec_attention_weights = dec_attention_weights_filled.reshape(shape)
#dec_self_attention_weights, dec_inter_attention_weights = \
    #dec_attention_weights.permute(1, 2, 3, 0, 4)
#
#Utility.check_shape(dec_self_attention_weights,
                #(num_blks, num_heads, data.num_steps, data.num_steps))
#Utility.check_shape(dec_inter_attention_weights,
                #(num_blks, num_heads, data.num_steps, data.num_steps))

#plt.show()

#valid_lens = torch.tensor([3, 2])
#encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
#encoder_blk.eval()
#Utility.check_shape(encoder_blk(X, valid_lens), X.shape)

#encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
#valid_lens = torch.tensor([3, 2])
#Utility.check_shape(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens), (2, 100, 24))

#decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
#X = torch.ones((2, 100, 24))
#state = [encoder_blk(X, valid_lens), valid_lens, [None]]
#Utility.check_shape(decoder_blk(X, state)[0], X.shape)



#add_norm = AddNorm(4, 0.5)
#shape = (2, 3, 4)
#Utility.check_shape(add_norm(torch.ones(shape), torch.ones(shape)), shape)
#ln = nn.LayerNorm(2)
#bn = nn.LazyBatchNorm1d()
#X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance for X in the training mode
#print('layer norm:', ln(X), '\nbatch norm:', bn(X))
#ffn = PositionWiseFFN(4, 8)
#ffn.eval()
#print(ffn(torch.ones((2, 3, 4)))[0])

#encoding_dim, num_steps = 32, 60
#pos_encoding = PositionalEncoding(encoding_dim, 0)
#X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
#P = pos_encoding.P[:, :X.shape[1], :]
#
#ProgressBoard.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         #figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
#
#plt.show()

#for i in range(8):
    #print(f'{i} in binary {i:>03b}')
#
#P = P[0, :, :].unsqueeze(0).unsqueeze(0)
#show_heatmap(P,
             #xlabel = 'Column (encoding dimension)',
             #ylabel = 'Row (position)',
             #figsize=(3.5, 4),
             #cmap='Blues')
#plt.show()

#num_hiddens, num_heads = 100, 5
#attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
#batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
#X = torch.ones((batch_size, num_queries, num_hiddens))
#Utility.check_shape(attention(X, X, X, valid_lens), (batch_size, num_queries, num_hiddens))
#num_hiddens, num_heads = 100, 5
#attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
#batch_size, num_queries, num_kvpairs = 2, 4, 6
#valid_lens = torch.tensor([3, 2])
#X = torch.ones((batch_size, num_queries, num_hiddens))
#Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
#Utility.check_shape(attention(X, Y, Y, valid_lens), (batch_size, num_queries, num_hiddens))

#data = MTFraEng(batch_size=128)
#embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
#encoder = Seq2SeqEncoder(len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
#decoder = Seq2SeqAttentionDecoder(len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
#model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)
#trainer = Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#
#engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
#fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
#
#preds,_ = model.predict_step(data.build(engs, fras), trainer.device(), data.num_steps)
#for en, fr, p in zip(engs, fras, preds):
    #translation = []
    #for token in data.tgt_vocab.to_tokens(p):
        #if token == '<eos>':
            #break
        #translation.append(token)
    #print(f'{en} => {translation}, bleu, {Utility.bleu(" ".join(translation), fr, k=2)}')
#
#_, dec_attention_weights = model.predict_step(data.build([engs[-1]], [fras[-1]]), Utility.try_gpu(), data.num_steps, True)
#attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weights], 0)
#attention_weights = attention_weights.reshape((1, 1, -1, data.num_steps))
#
#show_heatmap(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(), xlabel='Key positions', ylabel='Query positions')
#plt.show()

    
    
#vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
#batch_size, num_steps = 4, 7
#encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
#decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens, num_layers)
#X = torch.zeros((batch_size, num_steps), dtype=torch.long)
#state = decoder.init_state(encoder(X), None)
#output, state = decoder(X, state)
#Utility.check_shape(output, (batch_size, num_steps, vocab_size))
#Utility.check_shape(state[0], (batch_size, num_steps, num_hiddens))
#Utility.check_shape(state[1][0], (batch_size, num_hiddens))


#queries = torch.normal(0, 1, (2, 1, 20))
#keys = torch.normal(0, 1, (2, 10, 2))
#values = torch.normal(0, 1, (2, 10, 4))
#valid_lens = torch.tensor([2, 6])
#
#attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
#attention.eval()
#Utility.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
#show_heatmap(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
#plt.show()
#queries = torch.normal(0, 1, (2, 1, 2))
#keys = torch.normal(0, 1, (2, 10, 2))
#values = torch.normal(0, 1, (2, 10, 4))
#valid_lens = torch.tensor([2, 6])
#
#attention = DotProductAttention(dropout = 0.5)
#attention.eval()
#Utility.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
#show_heatmap(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
#plt.show()

#Q = torch.ones((2, 3, 4))
#K = torch.ones((2, 4, 6))
#Utility.check_shape(torch.bmm(Q, K), (2, 3, 6))
#print( Utility.masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])) )
#print( Utility.masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])) )
#def f(x):
    #return 2 * torch.sin(x) + x
#
#n = 40
#x_train, _ = torch.sort(torch.rand(n) * 5)
#y_train = f(x_train) + torch.randn(n)
#x_val = torch.arange(0, 5, 0.1)
#y_val = f(x_val)
#
##kernels = (Utility.gaussian, Utility.boxcar, Utility.constant, Utility.epanechnikov)
##names = ("Gaussian", "Boxcar", "Constant", "Epanechnikov")
#sigmas = (0.1, 0.2, 0.5, 1)
#kernels = [Utility.gaussian_with_width(sigma) for sigma in sigmas]
#names = ['Sigma ' + str(sigma) for sigma in sigmas]
##print(x_train)
##print(y_train)
##print(x_val)
##print(y_val)
#
#def plot(x_train, y_train, x_val, y_val, kernels, names, attention = False):
    #fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    #for kernel, name, ax in zip(kernels, names, axes):
        #y_hat, attention_w = Utility.nadaraya_watson(x_train, y_train, x_val, kernel)
        #if attention:
            #pcm = ax.imshow(attention_w.detach().numpy(), cmap='Reds')
        #else:
            #ax.plot(x_val, y_hat)
            #ax.plot(x_val, y_val, 'm--')
            #ax.plot(x_train, y_train, 'o', alpha=0.5)
        #ax.set_xlabel(name)
        #if not attention:
            #ax.legend(['y_hat', 'y'])
    #if attention:
        #fig.colorbar(pcm, ax=axes, shrink=0.7)
#
#plot(x_train, y_train, x_val, y_val, kernels, names)
#plt.show()
    
#fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
#
#x = torch.arange(-2.5, 2.5, 0.1)
#for kernel, name, ax in zip(kernels, names, axes):
    #ax.plot(x.detach().numpy(), kernel(x).detach().numpy())
    #ax.set_xlabel(name)
#plt.show()

#attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
#show_heatmap(attention_weights, xlabel = "Keys", ylabel="Queries")
#plt.show()
#data = MTFraEng(batch_size=128)
#embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
#encoder = Seq2SeqEncoder(len(data.src_vocab), embed_size, num_hiddens, num_layers)
#decoder = Seq2SeqDecoder(len(data.tgt_vocab), embed_size, num_hiddens, num_layers)
#model = Seq2Seq(encoder, decoder, tgt_pad = data.tgt_vocab['<pad>'], lr = 0.005)
#trainer = Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#
#engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
#fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
#
#preds,_ = model.predict_step(data.build(engs, fras), trainer.device(), data.num_steps)
#for en, fr, p in zip(engs, fras, preds):
    #translation = []
    #for token in data.tgt_vocab.to_tokens(p):
        #if token == '<eos>':
            #break
        #translation.append(token)
    #print(f'{en} => {translation}, bleu, {Utility.bleu(" ".join(translation), fr, k=2)}')

#data = MTRusEng(batch_size=128)
#embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
#encoder = Seq2SeqEncoder((len(data.src_vocab)), embed_size, num_hiddens, num_layers, dropout)
#
#decoder = Seq2SeqDecoder(len(data.tgt_vocab), embed_size, num_hiddens, num_layers)
#
#model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#
#
#
#engs = ['i see .', 'smile .', 'buy it .', 'i\'m ok.']
#fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
#
#
#preds,_ = model.predict_step(data.build(engs, fras), trainer.device(), data.num_steps)
#for en, fr, p in zip(engs, fras, preds):
    #translation = []
    #for token in data.tgt_vocab.to_tokens(p):
        #if token == '<eos>':
            #break
        #translation.append(token)
    #print(f'{en} => {translation}, bleu, {Utility.bleu(" ".join(translation), fr, k=2)}')
#
#
#
#
#
##src_tokens = torch.tensor([data.src_vocab[s] for s in src_sentence.split(" ")] + [data.src_vocab['<eos>']])
##enc_valid_len = torch.tensor([len(src_tokens)])
##tgt_tokens = torch.tensor(data.tgt_vocab['<bos>'])
#
#engs = ["Go"]
#rus = [""]
#preds, _ = model.predict_step(data.build(engs, rus), trainer.device(), data.num_steps)
#for en, p in zip(engs, preds):
    #translation = []
    #for token in data.tgt_vocab.to_tokens(p):
        #if token == '<eos>':
            #break
    #translation.append(token)
    #print(" ".join(translation))
    
#print( model.predict('it has hello world', data.src_vocab, data.num_steps, trainer.device()))  
#print(data.src_vocab["hello", "world"])
#print( model.predict_custom("hello world", data.src_vocab, data.tgt_vocab, 1, Utility.try_gpu()))

#engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
#fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
#preds, _ = model.predict_step(
    #data.build(engs, fas), Utility.try_gpu(), data.num_steps)
#for en, fr, p in zip(engs, fras, preds):
    #translation = []
    #for token in data.tgt_vocab.to_tokens(p):
        #if token == '<eos>':
            #break
        #translation.append(token)
    #print(f'{en} => {translation}, bleu,'
          #f'{Utility.bleu(" ".join(translation), fr, k=2):.3f}')





#vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
#batch_size, num_steps = 4, 9
#encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
#X = torch.zeros((batch_size, num_steps))
#enc_outputs, enc_state = encoder(X)
#Utility.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
#
#vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
#decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
#state = decoder.init_state(encoder(X))
#dec_outputs, state = decoder(X, state)
#Utility.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
#Utility.check_shape(state[1], (num_layers, batch_size, num_hiddens))

#data = MTRusEng(batch_size=3)
#src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
#print('source', src.type(torch.int32))
#print('decoder input', tgt.type(torch.int32))
#print('source len excluding pad:', src_valid_len.type(torch.int32))
#print('label:', label.type(torch.int32))
#
#print('source ', data.src_vocab.to_tokens(src[0].type(torch.int32)))
#print('target ', data.tgt_vocab.to_tokens(tgt[0].type(torch.int32)))

#data = MTRusEng()
#raw_text = data._download()
#src, tgt = data._tokenize(data._preprocess(raw_text))
#
#print(src[:8])
#print(tgt[:8])
#
#show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        #'count', src, tgt)
#plt.show()




#data = TimeMachine(batch_size = 1024, num_steps = 32)
#biGru = BiGRU(num_inputs = len(data.vocab), num_hiddens = 32)
#model = RNNLM(biGru, vocab_size = len(data.vocab), lr = 2)
#
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#
#print( model.predict('the time', 200, data.vocab, Utility.try_gpu()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers = 2)
#model = RNNLM(gru, vocab_size = len(data.vocab), lr=2)
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 200, data.vocab, Utility.try_gpu()) ) 
#data = TimeMachine(batch_size = 1024, num_steps = 32)
#rnn_block = StackedRNNScratch(num_inputs = len(data.vocab), num_hiddens = 32, num_layers = 2)
#model = RNNLMScratch(rnn_block, vocab_size = len(data.vocab), lr = 2)
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 20, data.vocab, Utility.try_gpu()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
#model = RNNLM(gru, vocab_size = len(data.vocab), lr=4)
#trainer = Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 20, data.vocab, Utility.try_gpu()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
#model = RNNLMScratch(gru, vocab_size = len(data.vocab), lr=4)
#trainer = Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 20, data.vocab, Utility.try_gpu()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
#model = RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
#trainer = Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 20, data.vocab, Utility.try_gpu()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
#model = RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
#trainer = Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 20, data.vocab, Utility.try_gpu()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#rnn = RNN(num_inputs = len(data.vocab), num_hiddens = 32)
#model = RNNLM(rnn, vocab_size=len(data.vocab), lr = 1)
#
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#print( model.predict('it has', 20, data.vocab, device=trainer.device()) ) 

#data = TimeMachine(batch_size = 1024, num_steps = 32)
#rnn = RNNScratch(num_inputs = len(data.vocab), num_hiddens = 32)
#model = RNNLMScratch(rnn, vocab_size = len(data.vocab), lr = 1)
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#
#print(model.predict('how to make one million dollars', 2000, data.vocab, Utility.try_gpu()))

#batch_size, num_inputs, num_hiddens, num_steps = 2, 15, 32, 100
#rnn = RNNScratch(num_inputs, num_hiddens)
#X = torch.ones((num_steps, batch_size, num_inputs))
#outputs, state = rnn(X)

#data = TimeMachine(1, 1)
#raw_text = data._download()
#print(raw_text[:60])

#data = Data()
#progress_board_plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6,3))
#plt.show()

#model = LinearRegression(lr = 0.1)
#trainer = Trainer(max_epochs=5)
#trainer.fit(model, data)

#onestep_preds = model(data.features).detach().numpy()
#progress_board_plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x', legend=['labels', '1-step preds'], figsize=(6, 3))

#multistep_preds = torch.zeros(data.T)
#multistep_preds[:] = data.x
#for i in range(data.num_train + data.tau, data.T):
    #multistep_preds[i] = model(multistep_preds[i - data.tau:i].reshape((1, -1)))
#multistep_preds = multistep_preds.detach().numpy()

#def k_step_pred(k):
    #features = []
    #for i in range(data.tau):
        #features.append(data.x[i : i+data.T-data.tau-k+1])
    #for i in range(k):
        #preds = model(torch.stack(features[i : i + data.tau], 1))
        #features.append(preds.reshape(-1))
    #return features[data.tau:]
#
#steps = (1, 4, 16, 64)
#preds = k_step_pred(steps[-1])
#plt.show()
#progress_board_plot(data.time[data.tau+steps[-1]-1:], [preds[k - 1].detach().numpy() for k in steps], 'time', 'x', legend=[f'{k}-step preds' for k in steps], figsize=(6,3))
#
#progress_board_plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         #[onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         #'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
#plt.show()
#
#data = TimeMachine(batch_size=1024, num_steps=32)
#rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
#model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
##rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
##model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
#trainer = Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
#trainer.fit(model, data)
#

#rnn = RNNScratch(num_inputs, num_hiddens)
#model = RNNLMScratch(rnn, vocab_size)
#NiN().layer_summary((1, 1, 224, 224))
#model = NiN(lr=0.05)
#trainer = Trainer(max_epochs=10)
#data = FashionMNIST(batch_size=128, resize=(224, 224))
#model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
#trainer.fit(model, data)

#model = GoogleNet(lr=0.01)
#trainer = Trainer(max_epochs=10, num_gpus=0)
#data = FashionMNIST(batch_size=128, resize=(96, 96))
#model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
#trainer.fit(model, data)

#data = Data()
#model = LinearRegression(lr=0.01)
#trainer = Trainer(max_epochs=5)
#trainer.fit(model, data)
#plt.show()
#
#onestep_preds = model(data.features).detach().numpy()
#progress_board_plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
                    #legend=['labels', '1-step-preds'], figsize=(6,3))
#
#multistep_preds = torch.zeros(data.T)
#multistep_preds[:] = data.x
#for i in range(data.num_train + data.tau, data.T):
    #multistep_preds[i] = model(
        #multistep_preds[i - data.tau:i].reshape((1, -1))
    #)
#multistep_preds = multistep_preds.detach().numpy()


#plt.show()

#data = TimeMachine(batch_size=2, num_steps=10)
#for X, Y in data.train_dataloader():
    #print('X:', X, '\nY:', Y)
    #break
    #
#X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
#H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
#
#print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
#
#print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))

#batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
#X = torch.ones((num_steps, batch_size, num_inputs))
#outputs, state = rnn(X)

#raw_text = data._download()
#text = data._preprocess(raw_text)
#words = text.split()
#vocab = Vocab(words)
#freqs = [freq for token, freq in vocab.token_freqs]
#
#bigram_tokens = ['---'.join(pair) for pair in zip(words[:-1], words[1:])]
#bigram_vocab = Vocab(bigram_tokens)
#
#trigram_tokens = ['--'.join(triple) for triple in
                  #zip(words[:-2], words[1:-1], words[2:])]
#trigram_vocab = Vocab(trigram_tokens)
#
#bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
#trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
#progress_board_plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         #ylabel='frequency: n(x)', xscale='log', yscale='log',
         #legend=['unigram', 'bigram', 'trigram'])
#plt.show()

#progress_board_plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
#plt.show()



#corpus, vocab = data.build(raw_text)
#print(len(corpus))
#print(len(vocab))

#text = data._preprocess(raw_text)
#tokens = data._tokenize(text)
#
#vocab = Vocab(tokens)
#indices = vocab[tokens[:10]]
#print('indices:', indices)
#print('words:', vocab.to_tokens(indices))

#print(','.join(tokens[:30]))
#def k_step_pred(k):
    #features = []
    #for i in range(data.tau):
        #features.append(data.x[i : i+data.T-data.tau-k+1])
    ## The (i+tau)-th element stores the (i+1)-step-ahead predictions
    #for i in range(k):
        #preds = model(torch.stack(features[i : i+data.tau], 1))
        #features.append(preds.reshape(-1))
    #return features[data.tau:]
#
#steps = (1, 4, 16, 64)
#preds = k_step_pred(steps[-1])
#progress_board_plot(data.time[data.tau+steps[-1]-1:],
         #[preds[k - 1].detach().numpy() for k in steps], 'time', 'x',
         #legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
#
#plt.show()

#model = LinearRegression(lr=0.01)
#trainer = Trainer(max_epochs=5)
#trainer.fit(model, data)
#plt.show()




#model = ResNet18()
#trainer = Trainer(max_epochs=10, num_gpus=1)
#data = FashionMNIST(batch_size=128, resize=(96, 96))
#model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
#trainer.fit(model, data)
#blk = Residual(6, use_1x1conv=True, strides=2)
#X = torch.randn(4, 3, 6, 6)
#print(blk(X).shape)
#model = BNLeNet(lr=0.01)
#trainer = Trainer(max_epochs=10, num_gpus=1)
#data = FashionMNIST(batch_size=128)
#model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
#trainer.fit(model, data)


#model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
#trainer = Trainer(max_epochs=10, num_gpus=1)
#data = FashionMNIST(batch_size=128, resize=(224, 224))
#
#trainer.fit(model, data)
#
#plt.show()
#model = AlexNet(lr=0.01)
#data = FashionMNIST(batch_size=128, resize=(224, 224))
#trainer = Trainer(max_epochs=10, num_gpus=1)
#trainer.fit(model, data)
#
#plt.show()

#AlexNet().layer_summary((1, 1, 224, 224))
#trainer = Trainer(max_epochs=10, num_gpus=1)
#data = FashionMNIST(batch_size=128)
#
#model = LeNet(lr=0.1)
#model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
#
#trainer.fit(model, data)
#
#plt.show()

#AlexNet().layer_summary((1, 1, 224, 224))
#model = AlexNet(lr = 0.01)
#data = FashionMNIST(batch_size=128, resize=(224, 224))
#trainer = Trainer(max_epochs=10)
#
#trainer.fit(model, data)
#
#plt.show()



#data = FashionMNIST(batch_size=256)
#model = SoftmaxRegression(num_outputs=10, lr=0.1)
#trainer = Trainer(max_epochs=10)
#trainer.fit(model, data)
#
#plt.show()

#X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
#
#
#pool2d = nn.MaxPool2d(3, padding=1, stride=2)
#print(pool2d(X))

#X = torch.normal(0, 1, (3, 3, 3))
#K = torch.normal(0, 1, (2, 3, 1, 1))
#Y1 = corr2d_multi_in_out_1x1(X, K)
#Y2 = corr2d_multi_in_out(X, K)
#print(X)
#print(K)
#print(Y1)
#print(Y2)
#assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

#X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               #[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
#K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
#K = torch.stack((K, K + 1, K + 2), 0)
#print(K)
#
#print(corr2d_multi_in_out(X, K))

#conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
#X = torch.rand((8, 8))
#Y = comp_conv2d(conv2d, X)
#print(Y.shape)
#K = torch.tensor([[1.0, -1.0]])
#
#Y = corr2d(X, K)
#
#conv2d = nn.LazyConv2d(1, kernel_size=(1,2), bias=False)
#
#X = X.reshape((1, 1, 6, 8))
#Y = Y.reshape((1, 1, 6, 7))
#print(X)
#print(Y)
#
#lr = 3e-2
#
#for i in range(10):
    #Y_hat = conv2d(X)
    #l = (Y_hat - Y) ** 2
    #conv2d.zero_grad()
    #l.sum().backward()
    #conv2d.weight.data[:] -= lr * conv2d.weight.grad
    #print(f'epoch {i + 1}, loss{l.sum():.3f}')
#
#
#print(conv2d.weight.data.reshape((1, 2)))
#
#print(conv2d.weight.data.reshape((1, 2)))
#def k_fold_data(data, k):
    #rets = []
    #fold_size = data.train.shape[0] // k
    #for j in range(k):
        #idx = range(j * fold_size, (j+1) * fold_size)
        #rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx), data.train.loc[idx]))
#
    #return rets
#
#
#def k_fold(trainer, data, k, lr):
    #val_loss, models = [], []
    #num_dropouts = [0]
    #weight_decays = [1e-4]
#
    #min_loss = None
    #min_param = None
#
#
    #for i, data_fold in enumerate(k_fold_data(data, k)):
        #model = KaggleHouseNueralNetwork(data.train.shape[1] // 16, 1e-4, lr)
        ##model = LinearRegression(lr)
        #model.board.yscale='log'
        #model.board.display = False
        #trainer.fit(model, data_fold)
#
        #val_loss.append(model.board.data['val_loss'][-1].y)
#
#
##
    #print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
#
    #return models




#data = KaggleHouse(batch_size=64)
#data.preprocess()
##
#trainer = Trainer(max_epochs=10)
#models = k_fold(trainer, data, k=5, lr=0.1)
#print(k_fold_data(data, 3))
#print(data.train.shape)
##
#plt.show()



#print(data.raw_train.iloc[:4, [0, 1, 2, 3,-3, -2, -1]])
#hparams = {'num_outputs':10,
#k
           #'num_hiddens_1':256,
           #'num_hiddens_2':256,
           #'dropout_1':0.5,
           #'dropout_2':0.5,
           #'lr':0.1}
#
#model = DropoutMLP(**hparams)
#data = FashionMNIST(batch_size=256)
#trainer = Trainer(max_epochs=10)
#trainer.fit(model, data)
#
#plt.show()

#model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
#data = FashionMNIST(batch_size=256)
#trainer = Trainer(max_epochs=10)
#trainer.fit(model, data)
#
#plt.show()


#
#
#X, y = next(iter(data.val_dataloader()))
#preds = model(X).argmax(axis=1)
#
#wrong = preds.type(y.dtype) != y
#X, y, preds = X[wrong], y[wrong], preds[wrong]
#labels = [a+'\n'+b for a, b in zip(data.text_labels(y), data.text_labels(preds))]
#data.visualize([X, y], labels=labels)

#data = FashionMNIST(resize=(32, 32))
#batch = next(iter(data.train_dataloader()))
#data.visualize(batch)
#
#X = torch.rand((2, 5))
#print(X)
#print(softmax(X))
#print(X.sum(0, keepdims=True))
#print(X.sum(1, keepdims=True))

#x = SoftmaxRegressionScratch(2, 3, 0.1)
#print(x.W)
#print(x.b)

#X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 60]])
#print(X.shape)


#
#tic = time.time()

#for X, y in data.train_dataloader():
    #continue
#print(f'{time.time() - tic:.2f} sec')


#data = DataSample(num_train=20, num_val=100, num_inputs=200, batch_size=5)
#trainer = Trainer(max_epochs=10)
#
#model = WeightDecay(wd = 3, lr = 0.01)
#trainer.fit(model, data)
#
#print('L2 Norm of w', model.get_w_b()[0])

#board = ProgressBoard('x')
#
#for x in np.arange(0, 10, 0.1):
    #board.draw(x, np.sin(x), 'sin', every_n = 2)
    #board.draw(x, np.cos(x), 'cos', every_n = 10)

#x = np.linspace(0, 10, 100)
#y = np.sin(x)
#
#plt.plot(x, y)
#
#plt.show()

#data = DataSample(num_train=20, num_val=100, num_inputs=200, batch_size=5)
#trainer = Trainer(max_epochs=10)
#
#def train_scratch(lambd):
    #model = WeightDecayScratch(num_inputs = 200, lambd = lambd, lr = 0.01)
#
    #trainer.fit(model, data)
    #print("L2 norm of 2:", float(model.l2_penalty(model.w)))
#
#train_scratch(0)



#print(model.w)
#print(model.b)

#X, y = next(iter(data.train_dataloader()))
#print('X shape : ', X.shape, "\ny shape: ", y.shape)
#print('features:', data.X[0],'\nlabel:', data.y[0])
