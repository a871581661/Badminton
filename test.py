
def acc_eval(data_loader,net,force_batch_num=None):

    counter = 0
    right = 0

    if force_batch_num is None:
        for batch_id, (batch, label) in enumerate(data_loader):
            b =batch.shape[0]
            if True in batch.isnan():
                continue
            batch = batch.float()
            out = net(batch)
            predict = out.max(dim=1).indices
            counter += b
            right += (predict == label).sum().item()
        acc = right / counter*100
        return acc



    else:
        for batch_id, (batch, label) in enumerate(data_loader):

            b = batch.shape[0]
            if True in batch.isnan():
                continue
            batch = batch.float()
            out = net(batch)
            predict = out.max(dim=1).indices
            counter += b
            right += (predict == label).sum().item()
        acc = right / counter * 100
        if batch_id>force_batch_num:
            return acc