import torch
from torch.nn import Module

class Post_Prob(Module):
    def __init__(self,fg_sigma,bg_sigma, c_size, stride, background_ratio, use_background,kernel_NN, device):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.fg_sigma = fg_sigma
        self.bg_sigma = bg_sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2 # 这里应该降采样了？
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background
        self.kernel_NN=kernel_NN
        if self.use_bg:
            self.bg_mat=torch.tensor([self.bg_sigma**2]).unsqueeze_(1).expand(1,(self.cood.size(1))**2)

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        dismat_points_per_image=[self.cal_dis_mat(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)
        all_points_dismat=torch.cat(dismat_points_per_image, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1) # 每个batch这里为1张的图片的所有点
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image) # 一张图每个点距离的集合（x-z）^2

            prob_list = []
            for dis, st_size,dismat in zip(dis_list, st_sizes,dismat_points_per_image):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        min_dis_index=torch.min(dis, dim=0, keepdim=True)[1]
                        dismat_softmax = (torch.pow(dismat,0.5) / (self.fg_sigma * 100)).gather(0,min_dis_index)
                        d = st_size * self.bg_ratio*dismat_softmax
                        bg_dis = (d - torch.sqrt(min_dis))**2
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * all_points_dismat)-torch.log(torch.pow(all_points_dismat,0.5))#self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list

    # cal KNN
    def cal_dis_mat(self,points):
        m, n = points.size(0), points.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(points, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(points, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(1, -2, points, points.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = (dist.clamp(min=1e-12).sqrt().sort(1)) # 计算距离矩阵并排序距离获得KNN
        dist = dist[0][:,1:self.kernel_NN+1] # 取固定数量的点
        dist =((torch.sum(dist,dim=1)/self.kernel_NN).clamp(max=100,min=10))*self.fg_sigma #计算距离并限定在100以内
        dist=torch.pow(dist, 2).unsqueeze_(1).expand(m,(self.cood.size(1))**2) #扩展成对每个点得矩阵
        if self.use_bg:
            dist=torch.cat([dist, self.bg_mat.cuda()], 0)
        return dist



