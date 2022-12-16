import numpy as np 
from datetime import datetime
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import os 
import matplotlib as plt
from tqdm import tqdm 

# faster-rcnn-lin-2.0 folder path 
DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

class FasterRcnn(): 

    def __init__(   self, 
                    num_classes: int = 21, 
                    device: str ='cpu' 
                ) -> None:
        
        self.device = device
        self.load_pretrained_model(num_classes)
        self.create_training_output_folder()
        

    def load_pretrained_model(self, num_classes: int ):
        """ we will be using Mask R-CNN, which is based on top of Faster R-CNN. 
            Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.
            Finetuning from a pretrained model
        
            Args: 
                num_classes: number of classes 
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        device = torch.device(self.device)
        model.to(device)
        self.model = model


    def gen_current_time_str(self, ): 
        now = datetime.now()
        now_string = now.strftime("%y%m%d-%H%M%S")

        return now_string

    def create_training_output_folder(self, ): 
        now_string = self.gen_current_time_str()
        exp_folder = DIR_PATH / 'output_model' / now_string
        if not exp_folder.is_dir(): 
            os.mkdir(exp_folder)
        self.exp_folder = exp_folder
        print(str(exp_folder) + ' has been created! ')

    def train(  self, 
                train_dataloader, 
                checkpoint_f=False, 
                lr: float = 0.005, 
                num_epochs: float = 10, 
                weight_decay: float = 0.005,
                momentum: float = 0.9, 
                epoch_num_ouputs: int = 1
            ): 
                
        params = [p for p in self.model.parameters() if p.requires_grad] 
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay) # パラメータ探索アルゴリズム
        
        # train from a existing checkpoint file
        if checkpoint_f: 
            checkpoint = torch.load(DIR_PATH / 'output_model' / checkpoint_f)
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch0 = checkpoint['epoch']
            loss0 = checkpoint['loss']
            losses_array = np.append(loss0, losses_array)
        
        else: 
            epoch0 = 0
            losses_array = np.array([])

        # turn the model into training mode
        self.model.train()

        print('training start')
        print('output file: ', self.exp_folder)

        for epoch in range(epoch0, num_epochs): 
            print( f'training at epoch{epoch}: ')
            for i, batch in enumerate(tqdm(train_dataloader)):
                
                image_batch, target_batch, image_id_batch = batch
                images = list(image.to(self.device) for image in image_batch)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in target_batch]

                try: 
                    loss_dict = self.model(images, targets)
                
                except ValueError:
                    print(targets['image_id'])
                    pass

                else: 
                    # there are 4 kinds of losses stored in loss_dict
                    # ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
                    # summation losses to get a total loss value
                    losses = sum(loss for loss in loss_dict.values())

                    # record the losses at each step
                    loss_value = losses.item()
                    losses_array = np.append(losses_array, loss_value)
                    
                    # clear gradient
                    optimizer.zero_grad()
                    
                    # Calculate gradients
                    losses.backward()

                    # Update parameters
                    optimizer.step()
                    
            print( f'------> training at epoch{epoch} completed; loss: {loss_value}')

            # save the general check point 
            if (epoch) % epoch_num_ouputs == 0: 
                checkpoint_name = 'checkpoint_epoch{:0>3}.pt'.format(epoch)
                checkpoint_save_infor = {'epoch': epoch+1,  
                                         'model_state_dict': self.model.state_dict(), 
                                         'optimizer_state_dict': optimizer.state_dict(),
                                         'loss': loss_value, }
                torch.save(checkpoint_save_infor, self.exp_folder / checkpoint_name)
                print('save the checkpoint: {}'.format(checkpoint_name))
            
    
        file_name = 'model.pt'
        torch.save(self.model, self.exp_folder / file_name)
        print('training completed; model saved to ', self.exp_folder / file_name)

    # def plot_loss(self): 
    #     plt.clf()
    #     ax = plt.subplot()
    #     ax.plot(self.train_loss)
    #     ax.set_title('train_loss')
    #     ax.set  
    

    def eval(self, weights=False): 
        """ predict from a fintuned model

            Args: 
                weights: trained model filename (e.g., 'train_ALL_VOC2007.cpu.pt')
        """
        if not weights: 
            self.weights = weights
            model_load_from_weights = torch.load(self.weights) 
            model_load_from_weights.to(self.device)
            return model_load_from_weights.eval()
        else: 
            return self.model.eval()

            

    

    