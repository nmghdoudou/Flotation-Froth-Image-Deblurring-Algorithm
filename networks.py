import  torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as py_models
import numpy
import copy


####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class Dis_content(nn.Module):
  def __init__(self):
    super(Dis_content, self).__init__()
    model = []
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
    model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(-1)
    outs = []
    outs.append(out)
    return outs

class MultiScaleDis(nn.Module):
  def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
    super(MultiScaleDis, self).__init__()
    ch = 64
    self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    self.Diss = nn.ModuleList()
    for _ in range(n_scale):
      self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
    tch = ch
    for _ in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
      tch *= 2
    if sn:
      model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
    else:
      model += [nn.Conv2d(tch, 1, 1, 1, 0)]
    return nn.Sequential(*model)

  def forward(self, x):
    outs = []
    for Dis in self.Diss:
      outs.append(Dis(x))
      x = self.downsample(x)
    return outs

class Dis(nn.Module):
  def __init__(self, input_dim, norm='None', sn=False):
    super(Dis, self).__init__()
    ch = 64
    n_layer = 6
    self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] #16
    tch = ch
    for i in range(1, n_layer-1):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] # 8
      tch *= 2
    model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
    tch *= 2
    if sn:
      model += [spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
    else:
      model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model.cuda(gpu)

  def forward(self, x_A):
    out_A = self.model(x_A)
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    return outs_A

####################################################################
#---------------------------- Encoders -----------------------------
####################################################################
class E_content(nn.Module):
  def __init__(self, input_dim_a, input_dim_b):
    super(E_content, self).__init__()
    encA_c = []
    tch = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
    self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
    self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
    self.conv5_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

    self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
    self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
    self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
    self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
    self.conv9_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
    self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
    self.conv10_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)

    self.conv11 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
    self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
    self.conv13 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
    self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
    self.conv14_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
    self.conv15 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
    self.conv15_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)

    self.share_ResBlock = INSResBlock(tch * 4, tch * 4)
    self.guasslayer = GaussianNoiseLayer()

  def forward(self, xa, xb):
      tmp_a1 = self.conv1(xa)

      x1_a = F.relu(self.conv2(tmp_a1))
      x2_a = self.conv3(x1_a)
      tmp_a2 = x2_a + tmp_a1  # residual link
      x3_a = F.relu(self.conv4(tmp_a2))
      x4_a = self.conv5(x3_a)
      tmp_a3 = x4_a + tmp_a2  # residual link
      x5_a = F.relu(self.conv4_1(tmp_a3))
      x6_a = self.conv5_1(x5_a)
      tmp_a4 = x6_a + tmp_a3


      tmp_a5 = self.conv6(tmp_a4)

      x7_a = F.relu(self.conv7(tmp_a5))
      x8_a = self.conv8(x7_a)
      tmp_a6 = x8_a + tmp_a5  # residual link
      x9_a = F.relu(self.conv9(tmp_a6))
      x10_a = self.conv10(x9_a)
      tmp_a7 = x10_a + tmp_a6  # residual link
      x11_a = F.relu(self.conv9_1(tmp_a7))
      x12_a = self.conv10_1(x11_a)
      tmp_a8 = x12_a + tmp_a7  # residual link

      tmp_a9 = self.conv11(tmp_a8)

      x13_a = F.relu(self.conv12(tmp_a9))
      x14_a = self.conv13(x13_a)
      tmp_a10 = x14_a + tmp_a9  # residual link
      x15_a = F.relu(self.conv14(tmp_a10))
      x16_a = self.conv15(x15_a)
      tmp_a11 = x16_a + tmp_a10  # residual link
      x17_a = F.relu(self.conv14_1(tmp_a11))
      x18_a = self.conv15_1(x17_a)
      x1_a = x18_a + tmp_a11  # residual link


      sh_block_a = self.share_ResBlock(x1_a)
      share_guass_a = self.guasslayer(sh_block_a)
      outputA = share_guass_a

      tmp_b1 = self.conv1(xb)

      x1_b = F.relu(self.conv2(tmp_b1))
      x2_b = self.conv3(x1_b)
      tmp_b2 = x2_b + tmp_b1  # residual link
      x3_b = F.relu(self.conv4(tmp_b2))
      x4_b = self.conv5(x3_b)
      tmp_b3 = x4_b + tmp_b2  # residual link
      x5_b = F.relu(self.conv4_1(tmp_b3))
      x6_b = self.conv5_1(x5_b)
      tmp_b4 = x6_b + tmp_b3

      tmp_b5 = self.conv6(tmp_b4)

      x7_b = F.relu(self.conv7(tmp_b5))
      x8_b = self.conv8(x7_b)
      tmp_b6 = x8_b + tmp_b5  # residual link
      x9_b = F.relu(self.conv9(tmp_b6))
      x10_b = self.conv10(x9_b)
      tmp_b7 = x10_b + tmp_b6  # residual link
      x11_b = F.relu(self.conv9_1(tmp_b7))
      x12_b = self.conv10_1(x11_b)
      tmp_b8 = x12_b + tmp_b7  # residual link

      tmp_b9 = self.conv11(tmp_b8)

      x13_b = F.relu(self.conv12(tmp_b9))
      x14_b = self.conv13(x13_b)
      tmp_b10 = x14_b + tmp_b9  # residual link
      x15_b = F.relu(self.conv14(tmp_b10))
      x16_b = self.conv15(x15_b)
      tmp_b11 = x16_b + tmp_b10  # residual link
      x17_b = F.relu(self.conv14_1(tmp_b11))
      x18_b = self.conv15_1(x17_b)
      x1_b = x18_b+ tmp_b11  # residual link

      sh_block_b = self.share_ResBlock(x1_b)
      share_guass_b = self.guasslayer(sh_block_b)
      outputB = share_guass_b

      return outputA,outputB

  def forward_b(self, xb):
      tmp_b1 = self.conv1(xb)

      x1_b = F.relu(self.conv2(tmp_b1))
      x2_b = self.conv3(x1_b)
      tmp_b2 = x2_b + tmp_b1  # residual link
      x3_b = F.relu(self.conv4(tmp_b2))
      x4_b = self.conv5(x3_b)
      tmp_b3 = x4_b + tmp_b2  # residual link
      x5_b = F.relu(self.conv4_1(tmp_b3))
      x6_b = self.conv5_1(x5_b)
      tmp_b4 = x6_b + tmp_b3

      tmp_b5 = self.conv6(tmp_b4)

      x7_b = F.relu(self.conv7(tmp_b5))
      x8_b = self.conv8(x7_b)
      tmp_b6 = x8_b + tmp_b5  # residual link
      x9_b = F.relu(self.conv9(tmp_b6))
      x10_b = self.conv10(x9_b)
      tmp_b7 = x10_b + tmp_b6  # residual link
      x11_b = F.relu(self.conv9_1(tmp_b7))
      x12_b = self.conv10_1(x11_b)
      tmp_b8 = x12_b + tmp_b7  # residual link

      tmp_b9 = self.conv11(tmp_b8)

      x13_b = F.relu(self.conv12(tmp_b9))
      x14_b = self.conv13(x13_b)
      tmp_b10 = x14_b + tmp_b9  # residual link
      x15_b = F.relu(self.conv14(tmp_b10))
      x16_b = self.conv15(x15_b)
      tmp_b11 = x16_b + tmp_b10  # residual link
      x17_b = F.relu(self.conv14_1(tmp_b11))
      x18_b = self.conv15_1(x17_b)
      x1_b = x18_b + tmp_b11  # residual link

      sh_block_b = self.share_ResBlock(x1_b)
      share_guass_b = self.guasslayer(sh_block_b)
      outputB = share_guass_b
      return outputB


"""
  def forward_a(self, xa):
    outputA = self.convA(xa)
    outputA = self.conv_share(outputA)
    return outputA

  def forward_b(self, xb):
    outputB = self.convB(xb)
    outputB = self.conv_share(outputB)
    return outputB
"""


class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)

        return self.gamma * out + input

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
      super(ChannelAttention, self).__init__()
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.max_pool = nn.AdaptiveMaxPool2d(1)

      self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
      self.relu1 = nn.ReLU()
      self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
      max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
      out = avg_out + max_out
      return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
      super(SpatialAttention, self).__init__()

      assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
      padding = 3 if kernel_size == 7 else 1

      self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      avg_out = torch.mean(x, dim=1, keepdim=True)
      max_out, _ = torch.max(x, dim=1, keepdim=True)
      x = torch.cat([avg_out, max_out], dim=1)
      x = self.conv1(x)
      return self.sigmoid(x)


class E_attr_concat(nn.Module):
  def __init__(self, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
    super(E_attr_concat, self).__init__()

    ndf = 64
    n_blocks=4
    max_ndf = 4

    self.refPad = nn.ReflectionPad2d(1)
    self.conv1 = nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)
    self.ca = ChannelAttention(ndf)
    self.sa = SpatialAttention()
    self.satt = selfattention(ndf)
    self.block1 = BasicBlock(ndf, ndf * 2, norm_layer, nl_layer)
    self.block2 = BasicBlock(ndf * 2, ndf * 3, norm_layer, nl_layer)
    self.block3 = BasicBlock(ndf * 3, ndf * 4, norm_layer, nl_layer)
    self.relu = nn.LeakyReLU(inplace=True)
    self.ca1 = ChannelAttention(ndf * 4)
    self.sa1 = SpatialAttention()
    self.satt1 = selfattention(ndf * 4)
    self.avgPol = nn.AdaptiveAvgPool2d(1)

    self.fc_B = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])
    self.fcVar_B = nn.Sequential(*[nn.Linear(ndf * 4, output_nc)])

  def forward(self, xb):

    x = self.refPad(xb)
    x = self.conv1(x)
    x = self.ca(x)*x
    x = self.sa(x)*x
    #x = self.satt(x) * x
    x= self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.relu(x)
    #x = self.ca1(x)*x
    #x = self.sa1(x)*x
    x = self.satt1(x) * x
    x_conv_B = self.avgPol(x)

    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_B, outputVar_B


  def forward_b(self, xb):
    x_conv_B = self.conv_B(xb)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_B, outputVar_B

####################################################################
#--------------------------- Generators ----------------------------
####################################################################

class G_concat(nn.Module):
  def __init__(self, output_dim_a, output_dim_b, nz):
    super(G_concat, self).__init__()
    self.nz = nz
    tch = 256
    dec_share = []
    dec_share += [INSResBlock(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 256 + self.nz
    decA1 = []
    for i in range(0, 3):
        decA1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decA2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch // 2
    tch = tch + self.nz
    decA3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch // 2
    tch = tch + self.nz
    decA4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
    self.decA1 = nn.Sequential(*decA1)
    self.decA2 = nn.Sequential(*[decA2])
    self.decA3 = nn.Sequential(*[decA3])
    self.decA4 = nn.Sequential(*decA4)

    tch = 256 + self.nz
    decB1 = []
    for i in range(0, 3):
        decB1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decB2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch // 2
    tch = tch + self.nz
    decB3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch // 2
    tch = tch + self.nz
    decB4 = [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
    self.decB1 = nn.Sequential(*decB1)
    self.decB2 = nn.Sequential(*[decB2])
    self.decB3 = nn.Sequential(*[decB3])
    self.decB4 = nn.Sequential(*decB4)


  def forward_D(self, x, z):
      out0 = self.dec_share(x)
      z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
      x_and_z = torch.cat([out0, z_img], 1)
      out1 = self.decA1(x_and_z)
      z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
      x_and_z2 = torch.cat([out1, z_img2], 1)
      out2 = self.decA2(x_and_z2)
      z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
      x_and_z3 = torch.cat([out2, z_img3], 1)
      out3 = self.decA3(x_and_z3)
      z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
      x_and_z4 = torch.cat([out3, z_img4], 1)
      out4 = self.decA4(x_and_z4)
      return out4

  def forward_B(self, x, z):
      out0 = self.dec_share(x)
      z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
      x_and_z = torch.cat([out0, z_img], 1)
      out1 = self.decB1(x_and_z)
      z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
      x_and_z2 = torch.cat([out1, z_img2], 1)
      out2 = self.decB2(x_and_z2)
      z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
      x_and_z3 = torch.cat([out2, z_img3], 1)
      out3 = self.decB3(x_and_z3)
      z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
      x_and_z4 = torch.cat([out3, z_img4], 1)
      out4 = self.decB4(x_and_z4)
      return out4

####################################################################
#--------------------------- losses ----------------------------
####################################################################
class PerceptualLoss():
    def __init__(self, loss, gpu=0, p_layer=14):
        super(PerceptualLoss, self).__init__()
        self.criterion = loss
        
        cnn = py_models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == p_layer:
                break
        self.contentFunc = model     

    def getloss(self, fakeIm, realIm):
        if isinstance(fakeIm, numpy.ndarray):
            fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
            realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
class PerceptualLoss16():
    def __init__(self, loss, gpu=0, p_layer=14):
        super(PerceptualLoss16, self).__init__()
        self.criterion = loss
#         conv_3_3_layer = 14
        checkpoint = torch.load('/vggface_path/VGGFace16.pth')
        vgg16 = py_models.vgg16(num_classes=2622)
        vgg16.load_state_dict(checkpoint['state_dict'])
        cnn = vgg16.features
        cnn = cnn.cuda()
#         cnn = cnn.to(gpu)
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
#             print(layer)
            model.add_module(str(i),layer)
            if i == p_layer:
                break
        self.contentFunc = model   
        del vgg16, cnn, checkpoint

    def getloss(self, fakeIm, realIm):
        if isinstance(fakeIm, numpy.ndarray):
            fakeIm = torch.from_numpy(fakeIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
            realIm = torch.from_numpy(realIm).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    
class GradientLoss():
    def __init__(self, loss, n_scale=3):
        super(GradientLoss, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.criterion = loss    
        self.n_scale = n_scale
        
    def grad_xy(self, img):
        gradient_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        gradient_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gradient_x, gradient_y

    def getloss(self, fakeIm, realIm):
        loss = 0
        for i in range(self.n_scale):
            fakeIm = self.downsample(fakeIm)
            realIm = self.downsample(realIm)
            grad_fx, grad_fy = self.grad_xy(fakeIm)
            grad_rx, grad_ry = self.grad_xy(realIm)            
            loss += pow(4,i) * self.criterion(grad_fx, grad_rx) + self.criterion(grad_fy, grad_ry)
        return loss  

class l1GradientLoss():
    def __init__(self, loss, n_scale=3):
        super(l1GradientLoss, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.criterion = loss    
        self.n_scale = n_scale
        
    def grad_xy(self, img):
        gradient_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        gradient_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gradient_x, gradient_y

    def getloss(self, fakeIm):
        loss = 0
        for i in range(self.n_scale):
            fakeIm = self.downsample(fakeIm)
            grad_fx, grad_fy = self.grad_xy(fakeIm)       
            loss += self.criterion(grad_fx, torch.zeros_like(grad_fx)) + self.criterion(grad_fy, torch.zeros_like(grad_fy))
        
        return loss   

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def meanpoolConv(inplanes, outplanes):
  sequence = []
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
  return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
  sequence = []
  sequence += conv3x3(inplanes, outplanes)
  sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
  return nn.Sequential(*sequence)

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer
def conv3x3(in_planes, out_planes):
  return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
  def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
    self.name = name
    self.dim = dim
    if n_power_iterations <= 0:
      raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
    self.n_power_iterations = n_power_iterations
    self.eps = eps
  def compute_weight(self, module):
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    weight_mat = weight
    if self.dim != 0:
      # permute dim to front
      weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    weight_mat = weight_mat.reshape(height, -1)
    with torch.no_grad():
      for _ in range(self.n_power_iterations):
        v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
        u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
    sigma = torch.dot(u, torch.matmul(weight_mat, v))
    weight = weight / sigma
    return weight, u
  def remove(self, module):
    weight = getattr(module, self.name)
    delattr(module, self.name)
    delattr(module, self.name + '_u')
    delattr(module, self.name + '_orig')
    module.register_parameter(self.name, torch.nn.Parameter(weight))
  def __call__(self, module, inputs):
    if module.training:
      weight, u = self.compute_weight(module)
      setattr(module, self.name, weight)
      setattr(module, self.name + '_u', u)
    else:
      r_g = getattr(module, self.name + '_orig').requires_grad
      getattr(module, self.name).detach_().requires_grad_(r_g)

  @staticmethod
  def apply(module, name, n_power_iterations, dim, eps):
    fn = SpectralNorm(name, n_power_iterations, dim, eps)
    weight = module._parameters[name]
    height = weight.size(dim)
    u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
    delattr(module, fn.name)
    module.register_parameter(fn.name + "_orig", weight)
    module.register_buffer(fn.name, weight.data)
    module.register_buffer(fn.name + "_u", u)
    module.register_forward_pre_hook(fn)
    return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
  if dim is None:
    if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
      dim = 1
    else:
      dim = 0
  SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
  return module

def remove_spectral_norm(module, name='weight'):
  for k, hook in module._forward_pre_hooks.items():
    if isinstance(hook, SpectralNorm) and hook.name == name:
      hook.remove(module)
      del module._forward_pre_hooks[k]
      return module
  raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

