import sys
sys.path.append('.')
sys.path.append('./nets')
sys.path.append('./trainer')
sys.path.append('./config')
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.augmentation import RandomBlur, RandomColorJitter, augCompose
from data.dataset import dataReadPip, loadedDataset, readIndex
from trainer.model_trainer import ModelTrainer
import argparse

def main(config_name):
    #import relevant modules
    module_config  = __import__(config_name)
    cfg = getattr(module_config, "Config")()
    module_network = __import__(cfg.network_name)
    
    # set global variables
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    if not os.path.exists(cfg.tensorboardlog):
        os.makedirs(cfg.tensorboardlog)
    exp_name=cfg.name

    #dataset loader
    if cfg.use_dataaug:
        data_tranform = [[RandomColorJitter, 0.5], [RandomBlur, 0.2]]
    else:
        data_tranform = None
    data_augment_op = augCompose(transforms=data_tranform)  # img augmentation of the dataset
    train_txt_process = dataReadPip(transforms=data_augment_op, resize_width=cfg.resize_width, enlarge=cfg.enlarge, center_crop_size=cfg.center_crop_size)
    val_txt_process  = dataReadPip(transforms=None, resize_width=cfg.resize_width, enlarge=cfg.enlarge, center_crop_size=cfg.center_crop_size)
    train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=False), preprocess=train_txt_process)
    val_dataset   = loadedDataset(readIndex(cfg.val_data_path, shuffle=False),   preprocess=val_txt_process)
    train_img_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,  num_workers=4, drop_last=True)
    val_img_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=cfg.val_batch_size,   shuffle=False, num_workers=4, drop_last=True)
    
    #build trainer
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()
    model = getattr(module_network, cfg.network_name)()
    if hasattr(model, 'load_from'):
        model.load_from(cfg)
    model = nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    model_trainer = ModelTrainer(model, cfg).to(device)

    # load pretrained model params if needed
    if cfg.pretrained_model:
        pretrained_dict = model_trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model_trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, "train info")
    
    # setup logger
    model_trainer.vis.log(cfg._state_dict(), "config")
    model_trainer.vis.log(model.state_dict(), "model params")
    model_trainer.vis.log(model_trainer.optimizer.state_dict(), "optimizer")
    twriter = SummaryWriter(cfg.tensorboardlog)

    try:  
        for epoch in range(0, cfg.epoch):
            #---------------training---------------#
            model_trainer.vis.log("training" + str(epoch), "train info")
            model.train()
            bar = tqdm(enumerate(train_img_loader), total=len(train_img_loader), leave=True, ncols=100, unit=str(" "+str(cfg.train_batch_size)+"ximg"))
            bar.set_description("Epoch %d ---Train--- " % epoch)
            if model_trainer.optimizer.state_dict()['param_groups'][0]["lr"] <= cfg.end_lr:
                model_trainer.optimizer.param_groups[0]["lr"]=cfg.end_lr
            for idx, (img, lab) in bar:
                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                pred = model_trainer.train_op(data, target)
                if not cfg.lr_decay_by_epoch and idx%cfg.vis_train_loss_every==0:
                    bar.write("learning rate:"+str(model_trainer.optimizer.state_dict()['param_groups'][0]['lr']))
                model_trainer.iters += 1
                if idx % cfg.vis_train_loss_every == 0:
                    model_trainer.vis.plot_many({
                        'train_total_loss': model_trainer.log_loss['total_loss'],
                        'train_output_loss': model_trainer.log_loss['output_loss'],
                        'train_fuse5_loss': model_trainer.log_loss['fuse5_loss'],
                        'train_fuse4_loss': model_trainer.log_loss['fuse4_loss'],
                        'train_fuse3_loss': model_trainer.log_loss['fuse3_loss'],
                        'train_fuse2_loss': model_trainer.log_loss['fuse2_loss'],
                        'train_fuse1_loss': model_trainer.log_loss['fuse1_loss'],
                    })

                    twriter.add_scalars(main_tag=exp_name+'/train_loss', tag_scalar_dict={
                        'train_total_loss': model_trainer.log_loss['total_loss'],
                        'train_output_loss': model_trainer.log_loss['output_loss'],
                        'train_fuse5_loss': model_trainer.log_loss['fuse5_loss'],
                        'train_fuse4_loss': model_trainer.log_loss['fuse4_loss'],
                        'train_fuse3_loss': model_trainer.log_loss['fuse3_loss'],
                        'train_fuse2_loss': model_trainer.log_loss['fuse2_loss'],
                        'train_fuse1_loss': model_trainer.log_loss['fuse1_loss'],
                    },global_step=model_trainer.iters)

                if idx % cfg.vis_train_acc_every == 0:
                    model_trainer.acc_op(pred[0], target)
                    model_trainer.vis.plot_many({
                        'train_pred_acc': model_trainer.log_acc['pred_acc'],
                        'train_pred_pos_acc': model_trainer.log_acc['pred_pos_acc'],
                        'train_pred_neg_acc': model_trainer.log_acc['pred_neg_acc'],
                        'learning_rate':      model_trainer.optimizer.state_dict()['param_groups'][0]['lr'],
                        'train_pred_ODS':     model_trainer.log_acc['pred_ODS'],
                        'train_pred_OIS':     model_trainer.log_acc['pred_OIS']
                    })

                    twriter.add_scalars(main_tag=exp_name+'/train_acc', tag_scalar_dict={
                        'train_pred_acc':     model_trainer.log_acc['pred_acc'],
                        'train_pred_pos_acc': model_trainer.log_acc['pred_pos_acc'],
                        'train_pred_neg_acc': model_trainer.log_acc['pred_neg_acc']
                    }, global_step=model_trainer.iters)

                    twriter.add_scalar(tag=exp_name+'/learning_rate', scalar_value=model_trainer.optimizer.state_dict()['param_groups'][0]['lr'], global_step=model_trainer.iters)

                    twriter.add_scalars(main_tag=exp_name+'/train_merits', tag_scalar_dict={
                        'train_pred_ODS':     model_trainer.log_acc['pred_ODS'],
                        'train_pred_OIS':     model_trainer.log_acc['pred_OIS']
                    }, global_step=model_trainer.iters)
                    if model_trainer.log_acc['pred_ODS'] > cfg.save_train_ods or model_trainer.log_acc['pred_OIS'] > cfg.save_train_ois:
                        model_trainer.log_acc['iters']=model_trainer.iters
                        model_trainer.vis.log(model_trainer.log_acc, "train_merites")
                        cfg.save_train_ods = model_trainer.log_acc['pred_ODS']
                        cfg.save_train_ois = model_trainer.log_acc['pred_OIS']
                    
                if idx % cfg.vis_train_img_every == 0:
                    model_trainer.vis.img_many({
                        'train_img':    data.cpu(),
                        'train_output': torch.sigmoid(pred[0].contiguous().cpu()),
                        'train_lab':    target.unsqueeze(1).cpu(),
                        'train_fuse5':  torch.sigmoid(pred[1].contiguous().cpu()),
                        'train_fuse4':  torch.sigmoid(pred[2].contiguous().cpu()),
                        'train_fuse3':  torch.sigmoid(pred[3].contiguous().cpu()),
                        'train_fuse2':  torch.sigmoid(pred[4].contiguous().cpu()),
                        'train_fuse1':  torch.sigmoid(pred[5].contiguous().cpu()),
                    })

                    twriter.add_images(tag=exp_name+'/train_img', img_tensor=data.cpu(), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_output', img_tensor=torch.sigmoid(pred[0].contiguous().cpu()), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_lab', img_tensor=target.unsqueeze(1).cpu(), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_fuse5', img_tensor=torch.sigmoid(pred[1].contiguous().cpu()), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_fuse4', img_tensor=torch.sigmoid(pred[2].contiguous().cpu()), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_fuse3', img_tensor=torch.sigmoid(pred[3].contiguous().cpu()), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_fuse2', img_tensor=torch.sigmoid(pred[4].contiguous().cpu()), global_step=model_trainer.iters)
                    twriter.add_images(tag=exp_name+'/train_fuse1', img_tensor=torch.sigmoid(pred[5].contiguous().cpu()), global_step=model_trainer.iters)
                
                if idx % cfg.vis_val_every == 0:
                    # -------------------- val ------------------- #
                    model.eval()
                    val_loss = {
                        'eval_total_loss':  0,
                        'eval_output_loss': 0,
                        'eval_fuse5_loss':  0,
                        'eval_fuse4_loss':  0,
                        'eval_fuse3_loss':  0,
                        'eval_fuse2_loss':  0,
                        'eval_fuse1_loss':  0,
                    }
                    val_acc = {
                        'pred_acc': 0,
                        'pred_pos_acc': 0,
                        'pred_neg_acc': 0,
                        'pred_ODS': 0,
                        'pred_OIS': 0,
                    }

                    bar.set_description("Epoch %d --- Evaluation --- " % epoch)

                    with torch.no_grad():
                        model_trainer.ois_merits = []
                        model_trainer.ods_merits = []
                        for idx, (img, lab) in enumerate(val_img_loader, start=1):
                            val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                            val_pred = model_trainer.val_op(val_data, val_target)
                            model_trainer.val_OIS_cal(val_pred[0], val_target)
                            model_trainer.val_ODS_cal(val_pred[0], val_target, thresh=cfg.acc_sigmoid_th)
                            model_trainer.acc_op(val_pred[0], val_target)
                            val_loss['eval_total_loss'] += model_trainer.log_loss['total_loss']
                            val_loss['eval_output_loss']+= model_trainer.log_loss['output_loss']
                            val_loss['eval_fuse5_loss'] += model_trainer.log_loss['fuse5_loss']
                            val_loss['eval_fuse4_loss'] += model_trainer.log_loss['fuse4_loss']
                            val_loss['eval_fuse3_loss'] += model_trainer.log_loss['fuse3_loss']
                            val_loss['eval_fuse2_loss'] += model_trainer.log_loss['fuse2_loss']
                            val_loss['eval_fuse1_loss'] += model_trainer.log_loss['fuse1_loss']
                            val_acc['pred_acc']         += model_trainer.log_acc['pred_acc']
                            val_acc['pred_pos_acc']     += model_trainer.log_acc['pred_pos_acc']
                            val_acc['pred_neg_acc']     += model_trainer.log_acc['pred_neg_acc']
                        else:
                            val_acc['pred_OIS'] = model_trainer.cal_ois_scores()
                            val_acc['pred_ODS'] = model_trainer.cal_ods_scores()
                            
                            model_trainer.vis.img_many({
                                'eval_img': val_data.cpu(),
                                'eval_output': torch.sigmoid(val_pred[0].contiguous().cpu()),
                                'eval_lab':    val_target.unsqueeze(1).cpu(),
                                'eval_fuse5':  torch.sigmoid(val_pred[1].contiguous().cpu()),
                                'eval_fuse4':  torch.sigmoid(val_pred[2].contiguous().cpu()),
                                'eval_fuse3':  torch.sigmoid(val_pred[3].contiguous().cpu()),
                                'eval_fuse2':  torch.sigmoid(val_pred[4].contiguous().cpu()),
                                'eval_fuse1':  torch.sigmoid(val_pred[5].contiguous().cpu()),
                            })

                            twriter.add_images(tag=exp_name+'/eval_img', img_tensor=val_data.cpu(), global_step=model_trainer.iters)
                            twriter.add_images(tag=exp_name+'/eval_output', img_tensor=torch.sigmoid(val_pred[0].contiguous().cpu()), global_step=model_trainer.iters)
                            twriter.add_images(tag=exp_name+'/eval_lab', img_tensor=val_target.unsqueeze(1).cpu(), global_step=model_trainer.iters)
                            twriter.add_images(tag=exp_name+'/eval_fuse5', img_tensor=torch.sigmoid(val_pred[1].contiguous().cpu()), global_step=model_trainer.iters)
                            twriter.add_images(tag=exp_name+'/eval_fuse4', img_tensor=torch.sigmoid(val_pred[2].contiguous().cpu()), global_step=model_trainer.iters)
                            twriter.add_images(tag=exp_name+'/eval_fuse3', img_tensor=torch.sigmoid(val_pred[3].contiguous().cpu()), global_step=model_trainer. iters)
                            twriter.add_images(tag=exp_name+'/eval_fuse2', img_tensor=torch.sigmoid(val_pred[4].contiguous().cpu()), global_step=model_trainer.iters)
                            twriter.add_images(tag=exp_name+'/eval_fuse1', img_tensor=torch.sigmoid(val_pred[5].contiguous().cpu()), global_step=model_trainer.iters)

                            model_trainer.vis.plot_many({
                                'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                'eval_output_loss':val_loss['eval_output_loss'] / idx,
                                'eval_fuse5_loss': val_loss['eval_fuse5_loss'] / idx,
                                'eval_fuse4_loss': val_loss['eval_fuse4_loss'] / idx,
                                'eval_fuse3_loss': val_loss['eval_fuse3_loss'] / idx,
                                'eval_fuse2_loss': val_loss['eval_fuse2_loss'] / idx,
                                'eval_fuse1_loss': val_loss['eval_fuse1_loss'] / idx,
                                'eval_pred_acc':     val_acc['pred_acc'] / idx,
                                'eval_pred_neg_acc': val_acc['pred_neg_acc'] / idx,
                                'eval_pred_pos_acc': val_acc['pred_pos_acc'] / idx,
                                'eval_pred_OIS':     val_acc['pred_OIS'],
                                'eval_pred_ODS':     val_acc['pred_ODS']
                            })

                            twriter.add_scalars(main_tag=exp_name+'/val_loss', tag_scalar_dict={
                                'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                'eval_output_loss':val_loss['eval_output_loss'] / idx,
                                'eval_fuse5_loss': val_loss['eval_fuse5_loss'] / idx,
                                'eval_fuse4_loss': val_loss['eval_fuse4_loss'] / idx,
                                'eval_fuse3_loss': val_loss['eval_fuse3_loss'] / idx,
                                'eval_fuse2_loss': val_loss['eval_fuse2_loss'] / idx,
                                'eval_fuse1_loss': val_loss['eval_fuse1_loss'] / idx,
                                'eval_pred_acc':     val_acc['pred_acc'] / idx,
                                'eval_pred_pos_acc': val_acc['pred_pos_acc'] / idx,
                                'eval_pred_neg_acc': val_acc['pred_neg_acc'] / idx,
                            }, global_step=model_trainer.iters)

                            twriter.add_scalars(main_tag=exp_name+'/val_merits', tag_scalar_dict={
                                'eval_pred_ODS':     val_acc['pred_ODS'],
                                'eval_pred_OIS':     val_acc['pred_OIS'],
                            }, global_step=model_trainer.iters)

                            # ----------------- save model ---------------- #
                            if cfg.save_ods < val_acc['pred_ODS'] / idx or cfg.save_ois < val_acc['pred_OIS']: 
                                save_acc = {k: v / idx for k, v in val_acc.items()}
                                save_acc["iters"]=model_trainer.iters
                                model_trainer.vis.log(save_acc, "val_merits")    
                                cfg.save_ods = (val_acc['pred_ODS'])
                                cfg.save_ois = (val_acc['pred_OIS'])   
                                if val_acc['pred_ODS'] / idx > cfg.save_condition or val_acc['pred_OIS'] > cfg.save_condition:
                                    model_trainer.saver.save(model, tag='%s_epoch(%d)_acc(%.5f_%.5f)' % (cfg.name, epoch, cfg.save_ods, cfg.save_ois))
                                    model_trainer.vis.log('Save Model %s_epoch(%d)_acc(%0.5f/%0.5f)' % (cfg.name, epoch, cfg.save_ods, cfg.save_ois), 'train info')
                    bar.set_description("Epoch %d --- Training --- " % epoch)
                    model.train()
            bar.write("Epoch{epo}:learning rate {lr}".format(epo=str(epoch), lr=str(model_trainer.optimizer.state_dict()['param_groups'][0]['lr'])))       
            if cfg.lr_decay_by_epoch:
                model_trainer.scheduler.step()
        twriter.close()
        bar.close()
    except KeyboardInterrupt:
        model_trainer.saver.save(model, tag='Auto_Save_Model')
        print("\n Catch KeyboardInterrupt, Auto Save final model: %s" % model_trainer.saver.show_save_pth_name)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="name of the training model")
    parser.add_argument("--name", type=str, required=True, help="the name of the model to be trained ")
    args = parser.parse_args()
    exp_net = args.name
    config_name = "config_" + exp_net
    main(config_name)
