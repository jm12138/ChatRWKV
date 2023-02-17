########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, os, gc
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.jit as jit


MyLayer = nn.Layer
def __nop(ob):
    return ob
MyFunction = __nop

if int(os.environ["RWKV_JIT_ON"]) > 0:
    MyLayer = jit.TranslatedLayer
    MyFunction = jit.to_static

print(f'\nRWKV_JIT_ON {os.environ["RWKV_JIT_ON"]}\n')

RWKV_RESCALE_LAYER = 6 # set x = x/2 every X layer (to avoid FP16 overflow)

############################################################################################################

class RWKV_RNN(MyLayer):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if args.FLOAT_MODE == 'fp32':
            self.FLOAT_MODE = paddle.float32
        elif args.FLOAT_MODE == 'fp16':
            self.FLOAT_MODE = paddle.float16
        elif args.FLOAT_MODE == 'bf16':
            self.FLOAT_MODE = paddle.bfloat16
        self.RUN_DEVICE = args.RUN_DEVICE

        with paddle.no_grad():
            w = paddle.load(args.MODEL_NAME + '.pdparams')
            args.n_embd = w['emb.weight'].shape[1]
            args.n_layer = 0
            keys = list(w.keys()) # refine weights and send to correct device
            print_need_newline = False
            for x in keys:
                w[x].stop_gradient = True
                if x == 'emb.weight' or 'ln0' in x:
                    continue
                
                block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, block_id+1)
                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
                    w[x] = w[x].t()
                
                if '.time_decay' in x:
                    w[x] = w[x].cast(self.FLOAT_MODE)
                    w[x] = -paddle.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].cast(self.FLOAT_MODE)
                else:
                    w[x] = w[x].cast(self.FLOAT_MODE)

                if args.FLOAT_MODE == 'fp16':
                    if 'att.output.weight' in x:
                        w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                    if 'ffn.value.weight' in x:
                        w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                
                if args.RUN_DEVICE == 'cuda':
                    w[x] = w[x].cuda()

                shape = w[x].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 1:
                    shape = f"  {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f"  {str(shape[0]).rjust(5)}      "
                if block_id == 0:
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    print(x.ljust(32), str(w[x].dtype).replace('torch.', '').ljust(10), shape)
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)
        print(f'\nn_layer {args.n_layer} n_embd {args.n_embd} ctx_len {args.ctx_len}')

        keys = list(w.keys()) # store weights in self.w
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        with paddle.no_grad(): # precompute embedding
            try:
                x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
            except:
                x = F.layer_norm(self.w.emb.weight.cast(self.FLOAT_MODE), (self.args.n_embd,), weight=self.w.blocks[0].ln0.weight.cast(self.FLOAT_MODE), bias=self.w.blocks[0].ln0.bias.cast(self.FLOAT_MODE))
            self.w.emb.weight = x.cast(self.FLOAT_MODE)

        self.eval()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def LN(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF_one(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = state[5*i+0].cast(self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x.cast(self.FLOAT_MODE)

        r = F.sigmoid(xr @ rw)
        k = paddle.square(F.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = paddle.concat((state[5*i+0].cast(self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x[-1,:].cast(self.FLOAT_MODE)

        r = F.sigmoid(xr @ rw)
        k = paddle.square(F.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    @MyFunction
    def SA_one(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = state[5*i+1].cast(self.FLOAT_MODE)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x.cast(self.FLOAT_MODE)

        r = F.sigmoid(xr @ rw)
        k = (xk @ kw).cast(self.FLOAT_MODE)
        v = (xv @ vw).cast(self.FLOAT_MODE)

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = paddle.maximum(pp, ww)
        e1 = paddle.exp(pp - p)
        e2 = paddle.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + time_decay
        p = paddle.maximum(ww, k)
        e1 = paddle.exp(ww - p)
        e2 = paddle.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = (a / b).cast(self.FLOAT_MODE)
        return (r * wkv) @ ow

    @MyFunction
    def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = paddle.concat((state[5*i+1].cast(self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x[-1,:].cast(self.FLOAT_MODE)

        r = F.sigmoid(xr @ rw)
        k = (xk @ kw).cast(self.FLOAT_MODE)
        v = (xv @ vw).cast(self.FLOAT_MODE)

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        T = x.shape[0]
        for t in range(T):
            ww = time_first + k[t]
            p = paddle.maximum(pp, ww)
            e1 = paddle.exp(pp - p)
            e2 = paddle.exp(ww - p)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = paddle.maximum(ww, k[t])
            e1 = paddle.exp(ww - p)
            e2 = paddle.exp(k[t] - p)
            if t != T - 1:
                aa = e1 * aa + e2 * v[t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[5*i+2] = e1 * aa + e2 * v[t]
                state[5*i+3] = e1 * bb + e2
                state[5*i+4] = p
            xx[t] = (a / b).cast(self.FLOAT_MODE)
        return (r * xx) @ ow

    def forward(self, tokens, state, preprocess_only = False):
        with paddle.no_grad():
            w = self.w
            args = self.args

            seq_mode = len(tokens) > 1

            x = w.emb.weight[tokens] if seq_mode else w.emb.weight[tokens[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()

            if state == None:
                state = paddle.zeros((args.n_layer * 5, args.n_embd))
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            SA = self.SA_seq if seq_mode else self.SA_one
            FF = self.FF_seq if seq_mode else self.FF_one

            for i in range(args.n_layer):
                ww = w.blocks[i].att
                x = x + SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].ffn
                x = x + FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                if args.FLOAT_MODE == 'fp16':
                    if (i+1) % RWKV_RESCALE_LAYER == 0:
                        x = x / 2

            if preprocess_only:
                return state

            x = self.LN(x[-1,:], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.cast(self.FLOAT_MODE), state
