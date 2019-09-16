# -*- coding:utf-8 -*-

# !/usr/bin/env python
'''
To process the ycb-video fit coco format for mask training
@author:hades_sun

''' 
import os 
import json
import cv2
import glob
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import math
import random
import colorsys
import mat4py as mat
import scipy.io as sio
from PIL import Image, ImageChops, ImageMath
import numpy as np
from skimage import measure,draw 
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from matplotlib import patches,  lines

class YCB_V(object):
    '''
    with ycb-video dataset,change it's format fit to coco format 
    @parameters:
        input:dataset path
        return:json file by coco annoatation format

    '''

    def __init__(self,dataset):

        # super(YCB_V2COCO,self).__init__()

        self.datapath=dataset
        
        self.path={}
       

        #get the file path
        self.getpath()
        

    def getpath(self):
        '''
        using the dateset path to get the whole data path to read data
        and format data
        
        '''
        #root path
        ab_path=self.datapath

        dir_list=glob.glob(ab_path+'/*')

        #models'path
        model_path=os.path.join(ab_path,"models/*")
        model_list = glob.glob(model_path)
        self.path["model_path"]=model_list
        dir_list.remove(ab_path+"/models")

        #images data path
        data_path=[]
        for dir in dir_list:
            if os.path.isdir(dir):
                data_path.append(dir)
            else:
                continue

        self.path['data_path']=data_path

    # def 


                
class COCO_M(object):
    '''
    transform the coco dataset format
    '''
    def __init__(self,path):
        #path: dict{"model_path":[...],"data_path":[...]}

        self.licenses=[{"name":'ycb-video',"id":1,"url":"219.223.170.155"}]
        self.info= {"author":"hades_sun"}
        self.categories=[]
        self.images=[]
        self.annoatations=[]
        self.path=path
        self.ann_id=0
        # self.ycb=ycb_v

    def getimg(self,path,img_name,id):
        #append image to images and get the w and h of the image
        path_img=os.path.join(path,img_name)
        img=cv2.imread(path_img)
        h,w=img.shape[:-1]
        dict_img= {"height": h,
              "flickr_url": "ycb-video",
              "license": 1,
              "id":id,
              "width": w,
              "date_captured": "2019/9/4",
              "file_name": img_name}
        self.images.append(dict_img)

              

    def getann_per_img(self,path_in,model_data,img_name,id,ann):
        #get annotation (mask and kpts) of one image
        
        #get mask and tranform to coco format
        #model_data(dict):{'1':model1,'2':model2,...}
        #
        path=os.path.join(path_in,img_name.split('-')[0]+'-'+'label.png')
        ann_id=self.ann_id
        mask_img=cv2.imread(path)
        img_shape=mask_img.shape[:-1]

        instances,mask_encode=self.mask_polygon(mask_img)
        _ids=list(instances.keys())
        

        cls_indexes=ann['cls_indexes'].flatten().tolist()
        poses=ann['poses']
        # print(poses.shape)
        cam_k=ann['intrinsic_matrix']
        cc=np.array([2]*8).reshape(8,1)
        
        for ii,item in enumerate(_ids):
            item_data=[]
            for ll in range(len(instances[item])):
                _x=instances[item][ll][:,0]
                _y=instances[item][ll][:,1]
                item_swap=np.c_[_y,_x]
                item_data.append(item_swap.flatten().tolist())
            
            dict_ann={}
            #category_id
            _id=int(item)

            pts=model_data[str(_id)].pts
            RT=poses[:,:,cls_indexes.index(_id)]
            _keypoints=self.getkpts(pts,RT,cam_k)
            keypoints=np.c_[_keypoints,cc]
            #
            rle=mask_encode[item]
            # mask=annToMask(rle)
            area=self.area(rle)
            bbox=self.anntobox(rle)

            dict_ann["area"]=float(area[0])
            dict_ann["bbox"]=bbox.flatten().tolist()
            dict_ann["category_id"]=_id
            dict_ann["keypoints"]=keypoints.flatten().tolist()
            dict_ann["iscrowd"]=0
            dict_ann["num_keypoints"]=8
            dict_ann["image_id"]=id
            dict_ann["id"]=ii+ann_id
            dict_ann["segmentation"]=item_data   

            self.annoatations.append(dict_ann)
        self.ann_id=ii+1+ann_id





    def getcateg(self):
        
        #get the objects name for structuring the categories
        obj_list=[]
        _categories=[]
        for dir in self.path['model_path']:
            if os.path.isdir(dir):
                _dir=dir.split('/')[-1]
                obj_list.append(_dir)
            else:
                print("error!")

        #sort the objects
        categories=sorted(obj_list,key=lambda x: int(x.split('_')[0]))
        # print(len(categories))


        for i,obj in enumerate(categories):
            dict_cate={"supercategory": "object",
                "keypoints": ["1", "2", "3", "4",
                            "5", "6", "7", "8",],
                "skeleton": [[1, 2], [1, 3], [2, 4], [1, 5], [5, 7], [7, 8],
                            [3, 4], [5, 6], [2, 6], [6, 8], [8, 4]]
                        }
            dict_cate["id"]=i+1
            dict_cate["name"]=obj
            _categories.append(dict_cate)
        self.categories=_categories
        # print(_categories)
        

    def mask_polygon(self,mask_image):
        '''
        mask image-->polygon for coco annotation
        '''
        h,w=mask_image.shape[:-1]

        img_np=np.array(mask_image[:,:,0])

        instance_segm={}
        instance_rle={}
        for _id in range(1,22):
            if _id in img_np:
                n_img=np.zeros((h,w),dtype=np.uint8)
                # n_img_encode=np.zeros((h,w,1),dtype=np.uint8)
                n_img[np.where(img_np==_id)]=1
                n_img_encode=n_img.reshape(h,w,1)

                n_img_encode=np.asfortranarray(n_img_encode)
                polygon=measure.find_contours(n_img,0.5)
                instance_segm["{}".format(_id)]=polygon
                instance_rle["{}".format(_id)]=maskUtils.encode(n_img_encode)
            else: 
                continue

        return instance_segm,instance_rle

        

    
    def annToRLE(self,segm,img_shape):
        h = img_shape[0]
        w = img_shape[1]
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rle)


        return [rle]
    def annToMask(self,rle):
        mask = maskUtils.decode(rle)
        return mask

    def area(self,rle):
        area = maskUtils.area(rle)
        return area
    
    def anntobox(self,rle):
        bbox = maskUtils.toBbox(rle)
        return bbox


    def save_json(self,save_path):
        #save data as a .json file
        
        json_dict={
            "licenses":self.licenses,
            "categories":self.categories,
            "images":self.images,
            "annotations":self.annoatations,
            "info":self.info

        }
        # json_dict={
        #     "licenses":self.licenses,
        #     "categories":self.categories,
        #     "images":self.images,
        #     "info":self.info

        # }


        json_path=os.path.join(save_path,'kpts_coco.json')
        with open (json_path,'w') as fp:
            json.dump(json_dict,fp)
            print("Done")
            fp.close()

        
    def getkpts(self,npts,RT,cam_k):
        #get the model's keypoints crossponding the train sample
        # npts=pts
        # RT=self.get_rt()
        # cam_K=self.get_cam()
        ##
        kpts=self.Corner3DBbox(npts) #return the 8 corners of cuboid
        
        #kpts=self.TransTo8P(xyz,w,h,d) #transform

        if len(kpts)>0:
            
            ##using the K and RT matrix to projection the 8 corners of cuboid to get keypoints
            #return kpts(nx2)           
            return self.prodjectionModel(cam_k,RT,kpts) 

        else:
            print("No points!")

        def get_rt(self):
            pass

        def get_cam(self):
            pass
        
    def prodjectionModel(self,K,R_t,pts):
        #projection the 3D model to 2D image
        #(u,v)=K*R_t*pts
        #return (u,v) the coordinates in 2D image

        _pts=np.ones((len(pts),4),dtype=np.float)
        _pts[:,:-1]=pts[:,:]
        uv=(K.dot(R_t)).dot(_pts.T)
        img_uv=np.zeros((len(pts),2),dtype=np.int)
        uv=uv.T

        img_uv[:,0],img_uv[:,1]=uv[:,0]/uv[:,2],uv[:,1]/uv[:,2]
        return img_uv

    def Corner3DBbox(self,pts):
        #from the points coordinates get 3d model 8 corners of object
        #return (x,y,z) the positive coordinate
        #return w,h,d

        x,y,z=np.max(pts,axis=0)

        x1,y1,z1=np.min(pts,axis=0)

        corners=np.array([[x,y,z],[x,y,z1],[x,y1,z],
                    [x,y1,z1],[x1,y,z],[x1,y,z1],
                    [x1,y1,z],[x1,y1,z1]])

        return corners

    def TransTo8P(self,xyz,w,h,d):
        #transform the xyz,w,h,d to coordinates of 8 corners 
        #return the 8 corners' coordinates 
        corners3D=np.zeros((8,3),dtype=np.float)
        x,y,z=xyz[0],xyz[1],xyz[2]
        corners3D[0,:]=x,y,z
        corners3D[1,:]=x,y,z-d
        corners3D[2,:]=x,y-h,z
        corners3D[3,:]=x,y-h,z-d
        corners3D[4,:]=x-w,y,z
        corners3D[5,:]=x-w,y,z-d
        corners3D[6,:]=x-w,y-h,z
        corners3D[7,:]=x-w,y-h,z-d

        return corners3D
           


class Model(object):
    '''
    get model info from special data format
    @parameters:
        inputs:model_file_path
        
    '''
    def __init__(self,path):

        self.path=path
        # self.name=obj_name
        self.pts=[] 
        self.loadpts()
        
        
        
    def loadpts(self):

        # path=self.path
        format=self.path.split('/')[-1].split('.')[-1]
        # print(format)
        if format=='obj':
            
            self.pts=self.readObjFile(self.path)

        elif format=='xyz':
            self.pts=self.readFile(self.path)

        elif format=='ply':
            self.pts=self.load_ply(self.path)

        else:
            print("Not support data format!")
            
    def getkpts(self,RT,cam_K):
        #get the model's keypoints crossponding the train sample
        npts=self.pts
        kpts=self.Corner3DBbox_p(npts) #find the coordinate and the w,h,d of 3d object model
        # kpts=self.TransTo8P(xyz,w,h,d) #transform

        if len(self.pts)>0:
            
            ##using the K and RT matrix to projection the 8 corners of cuboid to get keypoints
            #return kpts(nx2)           
            return prodjectionModel(cam_K,RT,kpts) 

        else:
            print("No points!")


    def readObjFile(self,file_path):
        #read 3d model file .obj file 
        #return the vertices of 3d model(Nx3):
        #
        vertex=[]
        fp=open(file_path,'r')

        line=fp.readline()

        line=line.rstrip('\n').rstrip('\r')
        i=1
        while True:
            line=str(line)
            if line.startswith('v') and not(line.startswith('vt')) and not(line.startswith('vn')):
                line=line.split()
                # if len(line)==4:
                vertex.append([line[1],line[2],line[3]])
                # i+=1
                # print(line,i)
                    

            elif line.startswith('vn') or line.startswith('vt') or line.startswith('s') \
                or line.startswith('f') or line.startswith('usemtl'):
                break
            else:
                line=fp.readline()
                line=line.rstrip('\n').rstrip('\r')
        # print(vertex)
        fp.close()
        vertex=np.array(vertex,dtype=np.float)

        return vertex

    def readFile(self,file_path):
        #read the .xyz file from file_path,and transfrom the file to a numpy
        #(Nx3)
        with open(file_path,'r') as fp:
            lines=fp.readlines()
            fp.close()
        pts=[]
        for line in lines:
            line=line.rstrip()
            line=line.split()
            pts.append(line)

        np_pts=np.array(pts,dtype=np.float)
        return np_pts

    def prodjectionModel(self,K,R_t,pts):
        #projection the 3D model to 2D image
        #(u,v)=K*R_t*pts
        #return (u,v) the coordinates in 2D image

        _pts=np.ones((len(pts),4),dtype=np.float)
        _pts[:,:-1]=pts[:,:]
        uv=(K.dot(R_t)).dot(_pts.T)
        img_uv=np.zeros((len(pts),2),dtype=np.int)
        uv=uv.T

        img_uv[:,0],img_uv[:,1]=uv[:,0]/uv[:,2],uv[:,1]/uv[:,2]
        return img_uv

    def Corner3DBbox(self,pts):
        #from the points coordinates get 3d model 8 corners of object
        #return (x,y,z) the positive coordinate
        #return w,h,d

        x,y,z=np.max(pts,axis=0)

        xm,ym,zm=np.min(pts,axis=0)


        #w
        w=x-xm
        #h
        h=y-ym
        #d
        d=z-zm

        return [x,y,y],w,h,d

    def TransTo8P(self,xyz,w,h,d):
        #transform the xyz,w,h,d to coordinates of 8 corners 
        #return the 8 corners' coordinates 
        corners3D=np.zeros((8,3),dtype=np.float)
        x,y,z=xyz[0],xyz[1],xyz[2]
        corners3D[0,:]=x,y,z
        corners3D[1,:]=x,y,z-d
        corners3D[2,:]=x,y-h,z
        corners3D[3,:]=x,y-h,z-d
        corners3D[4,:]=x-w,y,z
        corners3D[5,:]=x-w,y,z-d
        corners3D[6,:]=x-w,y-h,z
        corners3D[7,:]=x-w,y-h,z-d

        return corners3D

    def Corner3DBbox_p(self,pts):
        #from the points coordinates get 3d model 8 corners of object
        #return (x,y,z) the positive coordinate
        #return w,h,d

        x,y,z=np.max(pts,axis=0)

        x1,y1,z1=np.min(pts,axis=0)

        corners=np.array([[x,y,z],[x,y,z1],[x,y1,z],
                    [x,y1,z1],[x1,y,z],[x1,y,z1],
                    [x1,y1,z],[x1,y1,z1]])

        return corners

    def load_ply(self,path):
        """Loads a 3D mesh model from a PLY file.

        :param path: Path to a PLY file.
        :return: The loaded model given by a dictionary with items:
        - 'pts' (nx3 ndarray)
        - 'normals' (nx3 ndarray), optional
        - 'colors' (nx3 ndarray), optional
        - 'faces' (mx3 ndarray), optional
        - 'texture_uv' (nx2 ndarray), optional
        - 'texture_uv_face' (mx6 ndarray), optional
        - 'texture_file' (string), optional
        """
        f = open(path, 'r')

        # Only triangular faces are supported.
        face_n_corners = 3

        n_pts = 0
        n_faces = 0
        pt_props = []
        face_props = []
        is_binary = False
        header_vertex_section = False
        header_face_section = False
        texture_file = None

        # Read the header.
        while True:

            # Strip the newline character(s).
            line = f.readline().rstrip('\n').rstrip('\r')

            if line.startswith('comment TextureFile'):
                texture_file = line.split()[-1]
            elif line.startswith('element vertex'):
                n_pts = int(line.split()[-1])
                header_vertex_section = True
                header_face_section = False
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
                header_vertex_section = False
                header_face_section = True
            elif line.startswith('element'):  # Some other element.
                header_vertex_section = False
                header_face_section = False
            elif line.startswith('property') and header_vertex_section:
                # (name of the property, data type)
                pt_props.append((line.split()[-1], line.split()[-2]))
            elif line.startswith('property list') and header_face_section:
                elems = line.split()
                if elems[-1] == 'vertex_indices' or elems[-1] == 'vertex_index':
                    # (name of the property, data type)
                    face_props.append(('n_corners', elems[2]))
                    for i in range(face_n_corners):
                        face_props.append(('ind_' + str(i), elems[3]))
                elif elems[-1] == 'texcoord':
                    # (name of the property, data type)
                    face_props.append(('texcoord', elems[2]))
                    for i in range(face_n_corners * 2):
                        face_props.append(('texcoord_ind_' + str(i), elems[3]))
                else:
                    misc.log('Warning: Not supported face property: ' + elems[-1])
            elif line.startswith('format'):
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('end_header'):
                break

        # Prepare data structures.
        model = {}
        if texture_file is not None:
            model['texture_file'] = texture_file
        model['pts'] = np.zeros((n_pts, 3), np.float)
        if n_faces > 0:
            model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

        pt_props_names = [p[0] for p in pt_props]
        face_props_names = [p[0] for p in face_props]

        is_normal = False
        if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
            is_normal = True
            model['normals'] = np.zeros((n_pts, 3), np.float)

        is_color = False
        if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
            is_color = True
            model['colors'] = np.zeros((n_pts, 3), np.float)

        is_texture_pt = False
        if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
            is_texture_pt = True
            model['texture_uv'] = np.zeros((n_pts, 2), np.float)

        is_texture_face = False
        if {'texcoord'}.issubset(set(face_props_names)):
            is_texture_face = True
            model['texture_uv_face'] = np.zeros((n_faces, 6), np.float)

        # Formats for the binary case.
        formats = {
            'float': ('f', 4),
            'double': ('d', 8),
            'int': ('i', 4),
            'uchar': ('B', 1)
        }

        # Load vertices.
        for pt_id in range(n_pts):
            prop_vals = {}
            load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                        'red', 'green', 'blue', 'texture_u', 'texture_v']
            if is_binary:
                for prop in pt_props:
                    format = formats[prop[1]]
                    read_data = f.read(format[1])
                    val = struct.unpack(format[0], read_data)[0]
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = val
            else:
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(pt_props):
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = elems[prop_id]

            model['pts'][pt_id, 0] = float(prop_vals['x'])
            model['pts'][pt_id, 1] = float(prop_vals['y'])
            model['pts'][pt_id, 2] = float(prop_vals['z'])

            if is_normal:
                model['normals'][pt_id, 0] = float(prop_vals['nx'])
                model['normals'][pt_id, 1] = float(prop_vals['ny'])
                model['normals'][pt_id, 2] = float(prop_vals['nz'])

            if is_color:
                model['colors'][pt_id, 0] = float(prop_vals['red'])
                model['colors'][pt_id, 1] = float(prop_vals['green'])
                model['colors'][pt_id, 2] = float(prop_vals['blue'])

            if is_texture_pt:
                model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
                model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

        # Load faces.
        for face_id in range(n_faces):
            prop_vals = {}
            if is_binary:
                for prop in face_props:
                    format = formats[prop[1]]
                    val = struct.unpack(format[0], f.read(format[1]))[0]
                    if prop[0] == 'n_corners':
                        if val != face_n_corners:
                            raise ValueError('Only triangular faces are supported.')
                    elif prop[0] == 'texcoord':
                        if val != face_n_corners * 2:
                            raise ValueError('Wrong number of UV face coordinates.')
                    else:
                        prop_vals[prop[0]] = val
            else:
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(face_props):
                    if prop[0] == 'n_corners':
                        if int(elems[prop_id]) != face_n_corners:
                            raise ValueError('Only triangular faces are supported.')
                    elif prop[0] == 'texcoord':
                        if int(elems[prop_id]) != face_n_corners * 2:
                            raise ValueError('Wrong number of UV face coordinates.')
                    else:
                        prop_vals[prop[0]] = elems[prop_id]

            model['faces'][face_id, 0] = int(prop_vals['ind_0'])
            model['faces'][face_id, 1] = int(prop_vals['ind_1'])
            model['faces'][face_id, 2] = int(prop_vals['ind_2'])

            if is_texture_face:
                for i in range(6):
                    model['texture_uv_face'][face_id, i] = float(
                    prop_vals['texcoord_ind_{}'.format(i)])

        f.close()

        return model 


class DataPro(object):
    '''
    yvb-video dataset
    processing data files of the input path 
    '''

    def __init__(self,path):
        self.path=path
        self.imgs=[]
        self.depth=[]
        self.ann=[]
        self.masks=[]
        self.get_data()

    def get_data(self):
        if os.path.isdir(self.path):

            _f=glob.glob(self.path+'/*.mat')
            for f in _f:
                nf=f.split('/')[-1].split('-')[0]
                # print(f)
                ann=sio.loadmat(f)
                self.ann.append(ann) 
                self.imgs.append(nf+'-'+'color'+'.png')
                self.depth.append(nf+'-'+'depth'+'.png')
                self.masks.append(nf+'-'+'label'+'.png')

        else:
            print('The file path is empty!')



class Visualzation(object):
    '''
    for visualizing the coco dataset
    '''
    def __init__(self,coco_path,img_path):

        self.coco_path=coco_path

        self.coco_json=[]

        self.img_path=img_path

        self.dict_data=[]

        self.loadjson()
        
        self.color=np.array(
                    [[255, 255, 255],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255,0],
                    [255, 0, 255],
                    [0, 255, 255],
                [   128, 0, 0],
                    [0, 128, 0],
                    [0, 0, 128],
                    [128, 128, 0],
                    [128, 0, 128],
                    [0, 128, 128],
                    [64, 0, 0],
                    [0, 64, 0],
                    [0, 0, 64],
                    [64, 64, 0],
                    [64, 0, 64],
                    [0, 64, 64],
                    [192, 0, 0],
                    [0, 192, 0],
                    [0, 0, 192]],
                    dtype=np.uint8)

    def visual_all(self):
        #visualizing the all train data
        #one by one 

        pass
            
    def visual_img(self,_id):
        #for visualizing one img and 
        #annotation

        img_info,anns=self.get_info(_id)
        img_name=img_info[0]['file_name']
        print(img_name)

        #open the image
        img=cv2.imread(self.img_path+'/{}'.format(img_name))
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w=img.shape[:-1]

        #annotation
        ann_kpts=[]
        mask_img=[]
        for ann in anns:
            segm=ann['segmentation']
            rle=maskUtils.frPyObjects(segm, h, w)
            mask=maskUtils.decode(rle)
            #color
            cate_id=ann['category_id']
            color=self.color[cate_id]

            # print(mask.shape)

            coordinates=np.where(mask==1)

            x=coordinates[1].flatten().tolist()

            y=coordinates[0].flatten().tolist()

            _mask=np.zeros((h,w),dtype=np.uint8)

            _mask[x[:],y[:]]=1
            # print(_mask.min()
            mask_img.append(_mask)
            #keypoints
            kpts=ann['keypoints']
            np_kpts=np.array(kpts).reshape(int(len(kpts)/2),2)
            ann_kpts.append(np_kpts)

        # N=len(mask_img)
        self.visual_func(img,mask_img,ann_kpts)


    def visual_func(self,img,mask_img,ann_kpts):
        #implement the function of visualization
        

        N=len(mask_img)
        colors=random_colors(N)
        display_instances(img,N,mask_img)

        
            


    def get_info(self,id):
        #using the image id to get the image 
        #annotation info 
        # ids=[]
        img_info=[]
        
        for img in self.coco_json['images']:
            # print(img['id'])
            # ids.append(img['id'])

            if img['id']==id:
                img_info.append(img)
                break
            else:
                continue

        # print(ids)
        # print(id)
        if len(img_info)==0:
            print('No this id')
            return


        ann=[]

        for instance in self.coco_json['annotations']:
            if instance['image_id']==id:
                ann.append(instance)

        
        return img_info,ann




    def loadjson(self):
        #load the coco label json file 

        
        with open (self.coco_path,'r') as fp:
            json_file=json.load(fp)
            self.coco_json=json_file
            fp.close()
        ##


        





def visualize_img(img):
    #to visualize the img
    plt.figure("Image") # the name of window
    plt.imshow(img)
    plt.axis('on') # off can turn off the show of axis
    plt.title('image') # the title of window
    plt.show()



        
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                image[:, :, c] *
                                (1 - alpha) + alpha * color[c] * 255,
                                image[:, :, c])
    return image


    
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, N, masks,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    # N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1,figsize=(16, 16))
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        # if not np.any(kpts[i]):
        # #     # Skip this instance. Has no bbox. Likely lost in image cropping.
        #     continue


        # for ii in range(kpts[i].shape[0]):
        #     p=patches.Circle(kpts[i][ii],3,color=color)
        #     ax.add_patch(p)



        # if show_bbox:
        #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none')
        #     ax.add_patch(p)

        # # Label
        # if not captions:
        #     class_id = class_ids[i]
        #     score = scores[i] if scores is not None else None
        #     label = class_names[class_id]
        #     caption = "{} {:.3f}".format(label, score) if score else label
        # else:
        #     caption = captions[i]
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
