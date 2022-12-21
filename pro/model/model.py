import numpy as np 
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import os 
from typing import List, Dict
import matplotlib as plt
from torchvision.ops import nms
from tqdm import tqdm 
from PIL import ImageDraw, ImageFont, Image
from pro.util.utils import plot_results, gen_current_time_str, get_class_dict
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# faster-rcnn-lin-2.0 folder path 
DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

class FasterRcnn(): 

    def __init__(   self, 
                    class_name_dict: Dict , 
                    device: str ='cpu' 
                ) -> None:
        
        self.device = device
        self.exp_folder = ''
        self.class_name_dict = class_name_dict
        num_classes = len(class_name_dict) +1
        self.num_classes = num_classes
        
    
    def load_pretrained_model(self, num_classes):
        """ we will be using Mask R-CNN, which is based on top of Faster R-CNN. 
            Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.
            Finetuning from a pretrained model
        
            Args: 
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # num_classes = len(class_name_dict) +1
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        device = torch.device(self.device)
        model.to(device)
        self.model = model


    def create_training_output_folder(self, ): 
        now_string = gen_current_time_str()
        exp_folder = DIR_PATH / 'output_model' / now_string
        if not exp_folder.is_dir(): 
            os.mkdir(exp_folder)
            os.mkdir(exp_folder / 'checkpoint')
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

        self.load_pretrained_model(num_classes)

        
        # create a folder for storing the output of the training result
        if self.exp_folder == '': 
            self.create_training_output_folder() 
        
        
        # create the optimizer for training
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
            
    
        file_name = f'model.{self.device}.pt'
        torch.save(self.model, self.exp_folder / file_name)
        print('training completed; model saved to ', self.exp_folder / file_name)


    def eval(self, val_dataloader, load_from=False, plot_result=False, output_dir=None): 
        """ predict from a fintuned model

            Args: 
                load_from: trained model (including the relatively path), 
                (e.g., 'output_model/221216-175220/model.pt')
        """

        # load the model, turn on the eval mode of the model 
        if not load_from: 
            if len(self.exp_folder) != 0: 
                try: 
                    self.model.eval()
                    output_folder = self.exp_folder
                except NameError: 
                    print('No trained model is found; input the filename you want to load from (arg: load_from)')
            else: 
                raise Exception("Sorry, no numbers below zero")
                
        # load the model from the input argument
        else: 
            # model_path = Path(load_from).parent / 'model.pt' 
            model_path = Path(load_from)
            model_load_from_weights = torch.load(model_path, map_location=torch.device('cpu')) 
            model_load_from_weights.to(self.device)
            self.model = model_load_from_weights
            self.model.eval()
            

        # calculate the metrics
        # create a list for calculating mAP
        pred = [None]*val_dataloader.dataset.__len__()
        target_gt = [None]*val_dataloader.dataset.__len__()

        for i, (image, target, image_id) in enumerate(tqdm(val_dataloader.dataset)): 
            # image, target, image_id = next(iter(val_dataloader.dataset))

            # move the image arrays to the specified device one by one
            # images = list(img.to(device) for img in images)
            # turns image array into PIL Image module
            image = image.to(self.device)
            outputs = self.model([image])
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))

            # prediction 
            # using threshold 0.5 to filter out the boxes
            # apply nms on the prediction 
            boxes   = outputs[0]["boxes"].data.cpu().numpy()
            scores  = outputs[0]["scores"].data.cpu().numpy()
            labels  = outputs[0]["labels"].data.cpu().numpy()
            conf = 0.5
            boxes = boxes[scores >= conf].astype(np.int32)
            labels = labels[scores >= conf]
            scores = scores[scores >= conf]
            indices_keep = nms(torch.tensor(boxes.astype(float)),  
                    torch.tensor(scores.astype(float)), iou_threshold=0.5).cpu().numpy()

            # ground truth 
            boxes_gt = target["boxes"].data.cpu().numpy()
            labels_gt = target["labels"].data.cpu().numpy()
            image_gt = image.copy()
            
            pred[i] = dict( boxes=torch.tensor(boxes[indices_keep].astype(float)), 
                                scores=torch.tensor(scores[indices_keep].astype(float)), 
                                labels=torch.tensor(labels[indices_keep].astype(float))
                            )
            target_gt[i]  = dict( boxes=torch.tensor(boxes_gt.astype(float)), 
                                    labels=torch.tensor(labels_gt.astype(float))
                                )

            if plot_result : 
                # plot the predictions
                output_dir = plot_results(image, 
                        boxes[indices_keep], 
                        labels[indices_keep], 
                        image_id, 
                        self.class_name_dict,
                        scores=scores[indices_keep], 
                        gt=False,)

                # plot the ground truth
                plot_results(image_gt, 
                        boxes_gt, 
                        labels_gt, 
                        image_id, 
                        self.class_name_dict, 
                        gt=True,)        
    
        metric  = MeanAveragePrecision()
        metric.update(pred, target_gt)
        mAP = metric.compute()['map']
        print('mAP over {} images: {:4.3f}'.format(val_dataloader.dataset.__len__(), mAP) )
        print('images output to {}'.format(output_dir))



    

    