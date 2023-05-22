import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch import nn

from data_gen import train_loader, BATCH_SIZE, test_loader

# from model.test_Timenet import Model
from model.Informer import Model
from test import acc_eval
import configs





#model ini
seq_len=30
e_layers=2
top_k=3
d_model=128
d_ff=128
num_kernels=3
embed =63
dropout=.1

num_class=18





# Training settings
lr = 1e-3
batch_size = BATCH_SIZE  # 输入层维度  #每一批次训练多少个样例

dataset_frame = 30

# 循环帧数设定:
frames_ctl_list = [20, 10, 5]

for i in range(1, 3):
        net = Model(
            configs
        )

        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01, eps=1e-10)
        criterion = nn.CrossEntropyLoss()

        ############# Transformer_Train ###############
        # loss_list=[]
        # for epoch in range(5):
        #     for batch_id, (batch,label) in enumerate(train_loader):
        #         if True in batch.isnan():
        #             continue
        #         optimizer.zero_grad()
        #         batch=(batch[:,1:]-batch[:,:-1]).reshape(-1,29,63).to(torch.float32)
        #         out = net(batch)
        #         # shape: batch,step,hidden_size
        #         # out= out[:,-1,:]
        #         # shape: b,h
        #         loss=criterion(out,label.to(torch.long))
        #         loss.backward()
        #         optimizer.step()
        #         if (batch_id+1) %10 ==0:
        #             print('epoch:{},batchID:{},loss:{}'.format(epoch,batch_id+1,loss.item()))
        #         loss_list.append(loss.item())
        #
        # torch.save(net,'./model.pt')
        # np.save('./loss.npy',loss_list)

        loss_list = []
        test_acc_list = []
        train_acc_list = []


        # draw_control
        fig, ax = plt.subplots(1, 3)

        # 训练轮次
        epochs = 1
        # # 真正输入的帧数 = 帧数-1 (因为输入为fn-fn-1 因此会缺少1帧)
        # # real_input_frame = frame_ctl - 1
        # # 数据集起始帧+设定帧数=总帧数    batch[:,1+start_frame:]-batch[:,1+start_frame:-1] 保持偏移一帧
        # start_frame = dataset_frame - frame_ctl

        for epoch in range(epochs):
            for batch_id, (batch, label) in enumerate(train_loader):
                #### 100个batch对测试集进行测试#######
                if (batch_id) % 100 == 0:
                    with torch.no_grad():
                        test_acc=acc_eval(test_loader,net)
                        train_acc=acc_eval(train_loader,net,100)
                        print('testAcc:{:.3f},trainAcc:{:.3f}'.format(test_acc,train_acc))
                        test_acc_list.append(test_acc)
                        train_acc_list.append(train_acc)
                ### 100个batch对测试集进行测试#######

                if True in batch.isnan():
                    continue
                optimizer.zero_grad()
                batch = batch.float()
                out = net(batch)
                # shape: batch,step,hidden_size

                # shape: b,h
                loss = criterion(out, label.to(torch.long))
                loss.backward()
                optimizer.step()
                if (batch_id + 1) % 100 == 0:
                    print('epoch:{},batchID:{},loss:{}'.format(epoch, batch_id + 1, loss.item()))

                loss_list.append(loss.item())

                ax[0].plot(range(len(loss_list)), loss_list)
                ax[1].plot(range(len(test_acc_list)), test_acc_list)
                ax[2].plot(range(len(train_acc_list)), train_acc_list)
                plt.draw()
                plt.pause(.001)
                ax[0].cla()
                ax[1].cla()
                plt.cla()
print('high_trainacc:{:.3f},high_testacc:{:.3f}'.format(max(train_acc_list),max(test_acc_list)))
        # torch.save(net, './train/res/f{}_lstm_layer{}.pt'.format(frame_ctl, i))
        # np.save('./train/res/f{}_layer{}_loss.npy'.format(frame_ctl, i), loss_list)
        # np.save('./train/res/f{}_layer{}acc_list.npy'.format(frame_ctl, i), acc_list)

# nframe = np.loadtxt(file_path)[1:, 1:]
# nframe = nframe.reshape(-1, 21, 3)
#
# ####    绘图设置  #########
# arms = [6, 5, 4, 3, 19, 11, 12, 13, 14]
# legs = [10, 9, 8, 7, 2, 15, 16, 17, 18]
# body = [0, 1, 19, 20, 2]
#
# fig = plt.figure()
# ax_mas = fig.add_subplot(projection='3d')
# ax_mas.view_init(-60, 90)
#
# xmin = nframe[:, :, 0].min()
# xmax = nframe[:, :, 0].max()
# ymin = nframe[:, :, 1].min()
# ymax = nframe[:, :, 1].max()
# zmin = nframe[:, :, 2].min()
# zmax = nframe[:, :, 2].max()
# ####    绘图设置  #########
#
# for num in range(30, nframe.shape[0] - 30):
#     ax_mas.clear()
#     ax_mas.set_xlim(xmin, xmax)
#     ax_mas.set_ylim(ymin, ymax)
#     ax_mas.set_zlim(zmin, zmax)
#     frame = nframe[num]
#     batch = nframe[num - 30:num]
#     batch = (batch[1:] - batch[:-1]).reshape(29, 63)
#     batch = torch.from_numpy(batch).to(torch.float32) * 100
#
#     ax_mas.scatter(frame[:, 0], frame[:, 1], frame[:, 2])
#     ax_mas.plot(frame[arms, 0], frame[arms, 1], frame[arms, 2])
#     ax_mas.plot(frame[legs, 0], frame[legs, 1], frame[legs, 2])
#     ax_mas.plot(frame[body, 0], frame[body, 1], frame[body, 2])
#     plt.pause(0.05)
