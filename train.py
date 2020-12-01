import torch
from torch.optim import Adam
from SAE_resnext import SAE_DECODER
from pic_process import FaceData
from codecs import open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, batch_size):
    model_path = 'faceTSAE_model.pth'
    model = SAE_DECODER(encoder_grad=True)
    faceData = FaceData('train.txt', 'test.txt', batch_size)
    optim = Adam(model.parameters(), lr=1e-4)
    model.to(device)
    test_loss = 999
    total_cnt = 0
    for i in range(epoch):
        model.train()
        for j in range(faceData.tr_len//batch_size):
            x = faceData.next_train()
            optim.zero_grad()
            out = model(x)
            cost = model.loss(x, out)
            cost.backward()
            optim.step()
            step_cost = cost.detach().cpu().numpy()
            print('epoch', i, 'step', j, 'cost', step_cost)
            total_cnt += 1
            if total_cnt % 10000 == 0:
                model.eval()
                t_loss = 0
                t_cnt = 0
                for k in range(faceData.te_len//batch_size):
                    x = faceData.next_test()
                    out = model(x)
                    t_cost = model.loss(x, out)
                    ts_cost = t_cost.detach().cpu().numpy()
                    print('epoch', i, 'test step', k, 'test_cost', step_cost)
                    t_loss += ts_cost
                    t_cnt += 1
                t_loss /= t_cnt
                if t_loss <= test_loss:
                    test_loss = t_loss
                    torch.save(model.parameters(), model_path)
                open('tmp_record.txt', 'w', 'utf-8').write('test loss is '+str(t_loss)+' min loss is '+str(test_loss)+'\n')
                model.train()
        model.eval()
        total_loss = 0
        cnt = 0
        for j in range(faceData.te_len//batch_size):
            x = faceData.next_test()
            out = model(x)
            cost = model.loss(x, out)
            step_cost = cost.detach().cpu().numpy()
            print('epoch', i, 'test step', j, 'test_cost', step_cost)
            total_loss += step_cost
            cnt += 1
        total_loss /= cnt
        if total_loss <= test_loss:
            test_loss = total_loss
            torch.save(model.parameters(), model_path)
        open('tmp_record.txt', 'w', 'utf-8').write('test loss is '+str(total_loss)+' min loss is '+str(test_loss)+'\n')


if __name__ == "__main__":
    train(10, 64)