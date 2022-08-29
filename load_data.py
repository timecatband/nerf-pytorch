import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

IMAGE_WIDTH=100
IMAGE_HEIGHT=100


def load_bob_data(file, half_res=False, testskip=1):
    data = np.load(file)
    
    
    imgs = data["images"]
    poses = data["poses"]
    
    H, W = IMAGE_WIDTH, IMAGE_HEIGHT
    focal_length = 60 # ?

    imgs = torch.tensor(imgs)
    imgs = imgs.float()/255

    imgs = torch.permute(imgs, (0,3,1,2))
    imgs = F.interpolate(imgs, size=(100,100))
    imgs = torch.permute(imgs, (0,2,3,1))
    imgs = imgs.to(device)

    poses = torch.tensor(poses)    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

        
    return imgs, poses, render_poses, [H, W, focal], 63


