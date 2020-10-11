import torch
from torchvision import transforms
import cv2
import numpy as np

def get_reward(score, masked_score):
    reward = score - masked_score
    return np.clip(a=reward, a_min=0, a_max=0.99)

class Environment:
    """
    Environment class
    
    Description:
        Use a classifier for the original and masked image
        and its output difference to calculate the reward
    
    Definition:
        state: original image concatenated with the current mask
        action: x, y coordinate of the blob to mask
    
    Shapes:
        state: 4 x img_size x img_size
        action: 1 x 2
    
    Arguments:
        agent:          ddpg agent
        classifier:     classifier to get the confidence score
        img_dir:        path to the image to learn
        img_size:       image size
        blob_size:      the size of each blob to mask, the mask will be (x : x+blob_size, y : y+blob_size)
        done_threshold: termination condition, compare with the confidence score from the classifier
    """
    def __init__(self, agent, classifier, img_dir, img_size=320,
                 blob_size=5, done_threshold=0.5, device=torch.device('cuda:0')):
        self.agent,self.classifier,self.img,self.img_size,self.blob_size,self.done_threshold = \
            agent,classifier,cv2.imread(img_dir, cv2.IMREAD_COLOR),img_size,blob_size,done_threshold
        self.device = device

        self.reset()

    def reset(self):
        self.classifier.eval()
        self.classifier.to(self.device)

        self.img = cv2.resize(self.img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        self.mask = np.ones((self.img_size, self.img_size, 1))
        self.masked_img = self.mask*self.img

        self.score = self._get_confidence_score(img=self.masked_img)
        state = np.concatenate([self.img, self.mask], axis=2)
        return state

    def step(self, action):
        """
        action: 2x1 [x, y] - the coordinate of the to mask
        state: 4xHxW - 1st channel: current mask, other channels: RGB image
        """
        # get state
        x, y = action # range 0..1
        x, y = (x*self.img_size).astype(np.uint8), (y*self.img_size).astype(np.uint8) # convert to coordinates
        new_mask = np.ones((self.img_size, self.img_size, 1))
        new_mask[y:y+self.blob_size,x:x+self.blob_size] = 0
        self.mask[y:y+self.blob_size,x:x+self.blob_size] = 0
        self.masked_img = self.mask*self.masked_img
        state = np.concatenate([self.masked_img, self.mask], axis=2)

        # get reward
        masked_score = self._get_confidence_score(img=self.masked_img)

        reward = get_reward(score=self.score, masked_score=masked_score)

        # self.score = masked_score use if we get the difference from the original score or the previous score
        
        # get done
        done = masked_score < self.done_threshold

        # info
        info = {}

        return state, reward, done, info

    @torch.no_grad()
    def _get_confidence_score(self, img):
        def ToTensor3Channels(x):
            x = torch.from_numpy(x.astype(np.float32))
            x[...,:3] = x[...,:3]/255
            x = x.permute(2, 0, 1)
            return x

        preprocess = transforms.Compose([
            transforms.Lambda(ToTensor3Channels),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        x = preprocess(img).unsqueeze(0).float()
        score = self.classifier(x.to(self.device))[0,0].item()
        return score
