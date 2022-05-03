import numpy as np
from collections import namedtuple
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(X,mu,std):
    return (X - mu)/std

upper_limit, lower_limit = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None,mu=None,std=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta,mu,std))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta,mu,std)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta,mu,std)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta,mu,std)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def update_parameters(model_parameters,parameters_origin,delta_p_all,weights,model,dicts):
    #assert len(delta_p_all)==10
    model_parameters=list(model_parameters)
    parameters_origin=list(parameters_origin)
    for i, (k,v) in enumerate(model.named_parameters()):
    #for i,(p,q) in enumerate(zip(parameters_origin,model_parameters)):
        p=parameters_origin[i]
        q=model_parameters[i]
        temp=0
        for j,delta in enumerate(delta_p_all):
            temp+=delta[i].data*weights[j][dicts[k]]
        q.data=p.data+temp.detach()

def update_weights(grad,delta_p_all,weights,model,dicts,lr=0.01,momentum_buffer=0,momentum=False,alpha=0.9):
    for i,(key,v) in enumerate(model.named_parameters()):
        g=grad[i]
        ##g=g/g.norm()
        for j,delta in enumerate(delta_p_all):    
            grad_weight_j=torch.sum(delta[i]*g)
            if momentum:
                momentum_buffer[j][dicts[key]]=momentum_buffer[j][dicts[key]]*alpha+grad_weight_j
            else:
                momentum_buffer[j][dicts[key]]=grad_weight_j
                
            weights[j][dicts[key]]=weights[j][dicts[key]]-momentum_buffer[j][dicts[key]]*lr
    weights=torch.clamp(weights,0,1)
    return weights

def update_parameters_two(parameters,parameters1):
    for (p,q) in zip(parameters,parameters1):
        p.data=q.data
def model_difference(p1,p2):
    diff=0.0
    for (p,q) in zip(p1,p2):
        diff+=(p-q).sum().item()
    print("difference",diff)



def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model,mu,std):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for i,batch in enumerate(loader):
        input,_=batch['input'], batch['target']
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        #if normalize is None:
        #    model(input)
        #else:
        model(normalize(input,mu,std))
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def get_dicts(args,model):
    dicts={}
    if args.model=='PreActResNet18':
        if args.layer_wise==2:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=0
                elif "layer2" in k:
                    dicts[k]=0
                elif "layer3" in k:
                    dicts[k]=0
                elif "layer4" in k:
                    dicts[k]=0
                else:
                    dicts[k]=1
        elif args.layer_wise==3:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=1
                elif "layer2" in k:
                    dicts[k]=1
                elif "layer3" in k:
                    dicts[k]=1
                elif "layer4" in k:
                    dicts[k]=1
                else:
                    dicts[k]=2
        elif args.layer_wise==4:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=1
                elif "layer2" in k:
                    dicts[k]=1
                elif "layer3" in k:
                    dicts[k]=2
                elif "layer4" in k:
                    dicts[k]=2
                else:
                    dicts[k]=3
        elif args.layer_wise==5:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=0
                elif "layer2" in k:
                    dicts[k]=1
                elif "layer3" in k:
                    dicts[k]=2
                elif "layer4" in k:
                    dicts[k]=3
                else:
                    dicts[k]=4
        elif args.layer_wise==1:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=0
                elif "layer2" in k:
                    dicts[k]=0
                elif "layer3" in k:
                    dicts[k]=0
                elif "layer4" in k:
                    dicts[k]=0
                else:
                    dicts[k]=0
        elif args.layer_wise==6:
            for  k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "layer1" in k:
                    dicts[k]=1
                elif "layer2" in k:
                    dicts[k]=2
                elif "layer3" in k:
                    dicts[k]=3
                elif "layer4" in k:
                    dicts[k]=4
                else:
                    dicts[k]=5
    elif args.model=='WideResNet':
        if args.layer_wise==1:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=0
                elif "block3" in k:
                    dicts[k]=0
                else:
                    dicts[k]=0
        elif args.layer_wise==2:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=0
                elif "block3" in k:
                    dicts[k]=0
                else:
                    dicts[k]=1
        elif args.layer_wise==3:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=1
                elif "block3" in k:
                    dicts[k]=1
                else:
                    dicts[k]=2
        elif args.layer_wise==4:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=0
                elif "block2" in k:
                    dicts[k]=1
                elif "block3" in k:
                    dicts[k]=2
                else:
                    dicts[k]=3
        elif args.layer_wise==5:
            for k,v in model.named_parameters():
                if "module.conv1" in k:
                    dicts[k]=0
                elif "block1" in k:
                    dicts[k]=1
                elif "block2" in k:
                    dicts[k]=2
                elif "block3" in k:
                    dicts[k]=3
                else:
                    dicts[k]=4
    return dicts

class GAdaptor(object):
    def __init__(self,model,args,mu,std):
        super(GAdaptor,self).__init__()
        self.model_exploit=copy.deepcopy(model)
        self.args=args
        self.steps=0.0
        self.parameters_origin=None
        self.parameters_pre=None
        self.delta_p_all=[]
        #dicts={}
        self.dicts=get_dicts(args,model)
        print("dicts",self.dicts)
        self.best_meta_acc=None
        self.gap=args.gap
        self.num_models=args.num_gaps
        self.flag=True
        self.mu=mu
        self.std=std
        #self.epsilon = (args.epsilon / 255.)
        self.epsilon=args.epsilon
        self.pgd_alpha = (args.pgd_alpha / 255.)
        self.criterion = nn.CrossEntropyLoss()
        self.filename_best="./train_models/best_meta_reinit_"+str(args.reinitialize)+"_initType_"+str(args.initialize_type)+"_"+str(args.model)+"_"+args.dataset+"_gap_"+str(args.gap)+"_numgaps_"+str(args.num_gaps)+"_trainmodeepoch_"+str(args.train_mode_epoch)+"_momentum09_layerwise_"+str(args.layer_wise)+"_repeat_"+str(args.repeat)+"_MetaStart_"+str(args.MetaStartEpoch)+"_times_"+str(args.times)+"_"+args.meta_loss+".pt"
        self.filename_last="./train_models/last_meta_reinit_"+str(args.reinitialize)+"_initType_"+str(args.initialize_type)+"_"+str(args.model)+"_"+args.dataset+"_gap_"+str(args.gap)+"_numgaps_"+str(args.num_gaps)+"_trainmodeepoch_"+str(args.train_mode_epoch)+"_momentum09_layerwise_"+str(args.layer_wise)+"_repeat_"+str(args.repeat)+"_MetaStart_"+str(args.MetaStartEpoch)+"_times_"+str(args.times)+"_"+args.meta_loss+".pt"
    def take_step(self,epoch,model,opt,val_batches,train_batches,test_batches=None):
        #print("steps",self.steps)
        if epoch>=self.args.MetaStartEpoch:
            self.steps+=1
            if self.parameters_origin is None:
                self.parameters_origin=copy.deepcopy(list(model.parameters()))
                self.parameters_pre=copy.deepcopy(list(model.parameters()))
                self.model_exploit.load_state_dict(copy.deepcopy(model.state_dict()))
            else:
                model.eval()
                if self.steps%self.gap==0:
                    delta_p_temp=[]
                    for p_pre,p_cur in zip(self.parameters_pre,model.parameters()):
                        delta_p=(p_cur-p_pre).detach()
                        delta_p_temp.append(delta_p)
                    self.parameters_pre=copy.deepcopy(list(model.parameters()))       
                    self.delta_p_all.append(delta_p_temp)
                if self.steps%(self.gap*self.num_models)==0:
                    if self.args.initialize_type=='random':
                        variables=torch.rand((self.num_models,self.args.layer_wise),device=device)
                    elif self.args.initialize_type=='zero':
                        variables=torch.zeros((self.num_models,self.args.layer_wise),device=device)
                    elif self.args.initialize_type=='one':
                        variables=torch.ones((self.num_models,self.args.layer_wise),device=device)

                    self.model_exploit.load_state_dict(copy.deepcopy(model.state_dict()))
                        #print("before")
                        #model_difference(model.parameters(),model_exploit.parameters())
                    update_parameters(self.model_exploit.parameters(),self.parameters_origin,self.delta_p_all,variables,model,self.dicts)
                    #print("after")
                    ####################################Check the difference##################################
#                         for i,(p,q) in enumerate(zip(model_exploit.parameters(),model.parameters())):
#                             print("")                        
                    ###################################Check End##############################################
                    delta=0.0
                    if epoch>=self.args.train_mode_epoch:
                        self.model_exploit.train()
                    else:
                        self.model_exploit.eval()
                    print("#####################Optimize weights###################################")
                    for i in range(10):
                        val_n=0
                        val_robust_acc=0.0
                        val_robust_loss=0.0
                        momentum_buffer=torch.zeros_like(variables)
                        #bn_update(train_batches,model_exploit,20)
                        for j, batch in enumerate(val_batches):
                            X, y = batch['input'], batch['target']
                            # Random initialization
                            if self.args.attack == 'none':
                                delta = torch.zeros_like(X)
                            else:#
                                delta = attack_pgd(self.model_exploit, X, y, self.epsilon, self.pgd_alpha, self.args.attack_iters, self.args.restarts, self.args.norm, early_stop=self.args.eval,mu=self.mu,std=self.std)
                            delta = delta.detach()
                            
                            robust_output = self.model_exploit(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit),self.mu,self.std))
                            if self.args.meta_loss=='CE':
                                robust_loss = self.criterion(robust_output, y)
                            elif self.args.meta_loss=='kl':
                                output=self.model_exploit(normalize(X,self.mu,self.std))       
                                robust_loss=torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(robust_output,dim=1),F.softmax(output,dim=1))                                    

                            grad=torch.autograd.grad(robust_loss,self.model_exploit.parameters())
                            variables=update_weights(grad,self.delta_p_all,variables,model,self.dicts,lr=0.1,momentum_buffer=momentum_buffer,momentum=True,alpha=0.9)                                    
                            update_parameters(self.model_exploit.parameters(),self.parameters_origin,self.delta_p_all,variables,model,self.dicts)
                            val_robust_loss += robust_loss.item() * y.size(0)
                            val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                            val_n += y.size(0)
                        print("i",i,"validate robust acc:",val_robust_acc/val_n,"loss",val_robust_loss/val_n)
                    self.delta_p_all=[]
                    bn_update(train_batches,self.model_exploit,self.mu,self.std)
                    print("variables:",variables)
                    print("#####################Optimize weights End###################################")
                    if epoch>=150 and self.flag: 
                            self.gap=self.gap*self.args.times
                            self.flag=False
                    if test_batches is not None:
                        model.eval()
                        self.model_exploit.eval()
                        test_loss = 0
                        test_acc = 0
                        test_robust_loss = 0
                        test_robust_acc = 0
                        test_n = 0
                        test_acc_orgin=0.0
                        for i, batch in enumerate(test_batches):
                            X, y = batch['input'], batch['target']

                            # Random initialization
                            if self.args.attack == 'none':
                                delta = torch.zeros_like(X)
                            else:
                                delta = attack_pgd(self.model_exploit, X, y, self.epsilon, self.pgd_alpha, self.args.attack_iters, self.args.restarts, self.args.norm, early_stop=self.args.eval,mu=self.mu,std=self.std)
                            delta = delta.detach()

                            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit),self.mu,self.std))
                            out=self.model_exploit(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit),self.mu,self.std))
                            robust_loss = self.criterion(robust_output, y)

                            output = self.model_exploit(normalize(X,self.mu,self.std))
                            loss = self.criterion(output, y)

                            test_robust_loss += robust_loss.item() * y.size(0)
                            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()

                            test_acc_orgin+=(out.max(1)[1]==y).sum().item()

                            test_loss += loss.item() * y.size(0)
                            test_acc += (output.max(1)[1] == y).sum().item()
                            test_n += y.size(0)
                        print("meta adapted robust acc:", test_acc_orgin/test_n,"orginal robust acc:",test_robust_acc/test_n,"meta adapted clean acc:",test_acc/test_n)
                        if self.best_meta_acc is None:
                            self.best_meta_acc=test_acc_orgin/test_n
                        else:
                            if self.best_meta_acc<(test_acc_orgin/test_n):
                                self.best_meta_acc=test_acc_orgin/test_n
                                print("save best meta model at ", epoch )
                                if self.args.file_name is None:
                                    torch.save(self.model_exploit.state_dict(),self.filename_best)
                                else:
                                     torch.save(self.model_exploit.state_dict(),"./train_models/best_meta_"+self.args.file_name+".pt")
                        if self.args.file_name is None:             
                            torch.save(self.model_exploit.state_dict(),self.filename_last)
                        else:
                            torch.save(self.model_exploit.state_dict(),"./train_models/last_meta_"+self.args.file_name+".pt")
                            

                        
                    model.load_state_dict(copy.deepcopy(self.model_exploit.state_dict()))
                    self.parameters_origin=copy.deepcopy(list(self.model_exploit.parameters()))
                    self.parameters_pre=copy.deepcopy(list(self.model_exploit.parameters()))
                    model.train()
                    ###remove momentum buffer
                    opt_temp = torch.optim.SGD(model.parameters(), lr=opt.param_groups[0]['lr'], momentum=0.9, weight_decay=5e-4)
                    if self.args.reinitialize==1:
                        opt=opt_temp
                    else:
                        for i,(p,q) in enumerate(zip(opt.param_groups[0]['params'],opt_temp.param_groups[0]['params'])):
                            state=opt.state[p]
                            opt_temp.state[q]['momentum_buffer']=state['momentum_buffer']
                        opt=opt_temp
        return model,opt







