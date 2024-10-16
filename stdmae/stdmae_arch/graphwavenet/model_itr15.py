import numpy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torchdiffeq import odeint

class ChannelAttention2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("-----------显示x---ChannelAttention------------",x.shape)
        resdice = x


        avg_out = self.mlp(self.avg_pool(x))  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        max_out = self.mlp(self.max_pool(x))  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)


        out = avg_out + max_out
        # return self.sigmoid(out)
        out = resdice + out

        return self.sigmoid(out)

class SpatialAttention2(nn.Module):
    def __init__(self, kernel_size=(7,1)):
        super(SpatialAttention2, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(3,0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("-----------显示x----SpatialAttention-----------", x.shape)
        resdice = x
        print("---------显示--x-------",x.shape)

        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        print("---------显示--avg_out------", avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        print("---------显示--max_out------", max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        x = self.conv1(x)  # 通过卷积层得到注意力权重:(B,2,H,W)-->(B,1,H,W)
        x = x + resdice
        return self.sigmoid(x)


class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim, adjm):
        super(ODEFunc, self).__init__()

        print("----------adjm---xxx------",(np.array(adjm)).shape)
        self.x0 = 0
        self.alpha = nn.Parameter(0.8 * torch.ones(torch.tensor(adjm).shape[1]))
        print("----------self.alpha---xxx------",self.alpha.shape)
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        print("-----xxxx-----xx--self.w------",self.w.shape)
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        print("-----xxxx-----xx--self.d------", self.d.shape)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        print("-----xxxx-----xx--self.w2------", self.w2.shape)
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
        print("-----xxxx-----xx--self.d2------", self.d2.shape)

        # print("--------self.w2---------", self.w2.shape)
        # print("--------temporal_dim---------", temporal_dim)

    def forward(self, adj, x):
        print("-----------------------ODE调试开始--------------------")

        # print("-----------------检测到有没有执行---------------------------")
        print("----------------adj------------",adj.shape)
        print("----------------x------------", x.shape)

        print("----------------self.alpha------------", self.alpha.shape)



        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        print("---------------new-self.alpha--xxx----------", alpha.shape)

        print("---------x----------",x.shape)
        xa = torch.einsum('ij, kjlm->kilm', adj, x)

        print("---------------new-xa----------", xa.shape)

        # ensure the eigenvalues to be less than 1
        print("------------self.d---------",self.d.shape)
        d = torch.clamp(self.d, min=0, max=1)

        print("---------------self.w----------", self.w.shape)
        print("-----------xxx----self.w * d----------", (self.w * d).shape)
        print("-----------xxx----torch.t(self.w)----------", torch.t(self.w).shape)
        w = torch.mm(self.w * d, torch.t(self.w))
        print("-------------xxx--new-w---------", w.shape)

        print("---------------new-xa-1---------", x.shape)
        print("---------------new-xa-2---------", w.shape)
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        print("---------------self.d2----------", self.d2.shape)

        d2 = torch.clamp(self.d2, min=0, max=1)

        print("---------xxx-----d2----------", d2.shape)

        print("--------------self.w2--xxx--------", self.w2.shape)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))

        print("------------xxx---w2----------", w2.shape)


        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        print("---------------xw2----------", xw2.shape)

        # print("xw:", xw)
        # print("xw2:", xw2)
        # print("self.x0:", self.x0)  #这个没有
        # print("xa:", xa)
        # print("x:", x)
        print("------------alpha----------",alpha.shape)
        print("-----------xa---------",xa.shape)
        print("-----------xxxxxx---------", x.shape)
        print("-----------xw---------", xw.shape)
        print("-----------xw2---------", xw2.shape)
        print("-----------self.x0---------", self.x0)

        print("---------XXXX--alpha---------", alpha.shape)

        print("---------XXXX--self.x0---------", self.x0)
        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0

        print("---------------new-xa----------", f.shape)
        return f




class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):

        # 20240611项目
        # A = A.to(x.device)
        # 单元测试启用
        A = torch.tensor(A,dtype=torch.float)


        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphWaveNet_itr15(nn.Module):
    """
        Paper: Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
        Link: https://arxiv.org/abs/1906.00121
        Ref Official Code: https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
    """

    def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, **kwargs):

        super(GraphWaveNet_itr15, self).__init__()
        # print("----------num_nodes--------",num_nodes)
        # print("----------supports-------", type(supports))
        # print("----------supports-------", supports)
        # 20240612修改
        # ams = np.array(supports)
        # print("----------supports-------", ams.size)
        #
        # print("----------supports00-------", ams[0].shape)
        # print("----------supports11-------", ams[1].shape)

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        # self.odefun0 = nn.ModuleList()
        self.odefun1 = nn.ModuleList()
        # self.fc_his_t = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        # self.fc_his_s = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        # print("-----num_nodes----",num_nodes)
        self.cc = ChannelAttention2(in_planes=num_nodes,ratio=16)
        self.ss = SpatialAttention2(kernel_size=(7,1))


        print("-------in_dim-in_dim-in_dim-----------",in_dim)
        print("-------residual_channels-residual_channels-residual_channels-----------",residual_channels)
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1
        # print("---------supports-----------",type(supports))
        # print("---------supports-----------",torch.tensor(supports)[0].shape)
        # odexm = [3, 4, 6, 7, 9, 10, 12, 13]
        # odexm0 = [13, 12, 10, 9, 7, 6, 4, 3]
        odexm1 = [12,10,9,7,6,4,3,1]
        # self.odefunG = ODEFunc(residual_channels, 1, supports[0])
        print("----------supports--------------",len(supports))
        print("----------supports[0]--------------", len(supports[0]))
        print("----------supports[0]-xxx-------------", (np.array(supports[0])).shape)
        print("----------supports[1]--------------", len(supports[1]))
        print("----------supports[1]-xxx-------------", (np.array(supports[0])).shape)
        self.odefunG = ODEFunc(256, 1, supports[0])
        self.conv = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=1)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            # 20240617新增循环结构
            # self.odefun0.append(ODEFunc(residual_channels, odexm0[2 * b], supports[0]))
            # self.odefun0.append(ODEFunc(residual_channels, odexm0[2 * b + 1], supports[0]))
            self.odefun1.append(ODEFunc(residual_channels, odexm1[2 * b], supports[0]))
            self.odefun1.append(ODEFunc(residual_channels, odexm1[2 * b + 1], supports[0]))

            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        # print()
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.receptive_field = receptive_field



    def forward(self, input, hidden_states):
        """feed forward of Graph WaveNet.
        Args:
            input (torch.Tensor): input history MTS with shape [B, L, N, C].
            His (torch.Tensor): the output of TSFormer of the last patch (segment) with shape [B, N, d].
        Returns:
            torch.Tensor: prediction with shape [B, N, L]
        """

        # reshape input: [B, L, N, C] -> [B, C, N, L]

        # print("-----------input----------",input.shape)
        # print("-----------------------")
        input = input.transpose(1, 3)

        print("------------问题1-xxx------------", input.shape)

        # feed forward
        input = nn.functional.pad(input,(1,0,0,0))


        input = input[:, :2, :, :]

        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        print("------------问题1-------------",x.shape)
        x = self.start_conv(x)
        print("------------问题2-------------", x.shape)

        print("------------------x-xxxx---------------",x.shape)
        skip = 0


        # calculate the current adaptive adj matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        adj_supports1 = torch.tensor(new_supports[0]).to(device)
        adj_supports2 = torch.tensor(new_supports[1]).to(device)
        adj_supports3 = new_supports[2].to(device)

        adj = adj_supports1 + adj_supports2 + adj_supports3


        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1 skip_convs
            #                                          |
            # ---------------------------------------> + ------------->	*skip*  输出

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)

            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            print("-----------x---filter * gate-----------",x.shape)

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            odexgconv = x.transpose(1, 2)

            odexgconv = odexgconv.transpose(2, 3)
            odexgconvs = self.odefun1[i](adj,odexgconv)
            odexgconvs = odexgconvs.transpose(1, 2)
            odexgconvs = odexgconvs.transpose(1, 3)
            x = odexgconvs
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        # adj = adj
        # print("-----------adj-----------",adj.shape)

        # 修改前区域
        # hidden_states_t = self.fc_his_t(hidden_states[:,:,:96])        # B, N, D
        # hidden_states_t = hidden_states_t.transpose(1, 2).unsqueeze(-1)
        # skip = skip + hidden_states_t
        # hidden_states_s = self.fc_his_s(hidden_states[:,:,96:])        # B, N, D
        # hidden_states_s = hidden_states_s.transpose(1, 2).unsqueeze(-1)
        # skip = skip + hidden_states_s



        
        # x = F.relu(skip)
        #
        # # print("------------x---x------------",x.shape)
        # odexg = x.transpose(1, 2)
        # odexg = odexg.transpose(2, 3)
        # # print("---------odexg-----------",odexg.shape)
        # # odexgo = self.odefunG(odexg,adj)
        # odexgo = self.odefunG(adj, odexg)
        # odexgo = odexgo.transpose(1, 2)
        # odexgo = odexgo.transpose(1, 3)
        # x = x + odexgo

        print("-------------------skip--------------------",skip.shape)
        # print("-------------xx-skip--------------",skip.shape)
        hidden_states = hidden_states[:,:,:96].unsqueeze(-1)
        print("------hidden_states------",hidden_states.shape)
        hidden_states_t = self.cc(hidden_states).transpose(1, 2)
        hidden_states_t = self.conv(hidden_states_t)

        # print("--------------------hidden_states_t--------------",hidden_states_t.shape)
        hidden_states_s = self.ss(hidden_states).transpose(1, 2)
        hidden_states_s = self.conv(hidden_states_s)
        # print("--------------------hidden_states_s--------------", hidden_states_s.shape)
        # print("-------------显示---------------")
        # print(skip.shape)
        skip = skip + hidden_states_t
        skip = skip + hidden_states_s
        x = F.relu(skip)


        # x = self.odefun()

        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        return x


if __name__=='__main__':
    input = torch.rand(1, 12, 307, 2)
    hidden_states = torch.rand(1, 307, 192)
    matrix = [[0 for _ in range(307)] for _ in range(307)]
    supports = [matrix,matrix]
    # supports = torch.tensor(supports)
    print("-----------x---xxxxxxxx-xxx---------")
    # print("-----------supports---------",supports[0].shape)
    print(type(supports))

    net = GraphWaveNet_itr15(num_nodes=307,supports=supports)
    print(net)

    x = net(input,hidden_states)
    print(x.shape)


    pass