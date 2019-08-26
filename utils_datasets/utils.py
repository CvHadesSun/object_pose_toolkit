
import mat4py as mat

import scipy.io as sio

import matplotlib.pyplot as plt

from PIL import Image, ImageChops, ImageMath

import numpy as np

# ann=mat.loadmat('000001-meta.mat')

def Visual3DModel(pts,bbox):

    #visualize the 3D model of a object by points cloud
    #
    ax = plt.subplot(111, projection='3d')
    X,Y,Z = pts[:,0],pts[:,1],pts[:,2]
    x,y,z=bbox[:,0],bbox[:,1],bbox[:,2]

    ax.scatter(X, Y, Z, c='g')
    ax.scatter(x, y, z, c='r',s=30)
    #label the axis name
    ax.set_zlabel('Z')  
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()

def visualize_img(img):
    #to visualize the img
    plt.figure("Image") # the name of window
    plt.imshow(img)
    plt.axis('on') # off can turn off the show of axis
    plt.title('image') # the title of window
    plt.show()

def Visual2dImg(uv,img):
    #plot scatter figure by uv coordinates
    x,y=uv[:,0],uv[:,1]
    # img=np.zeros((480,640,3),dtype=np.int)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #
    ax1.set_title('Scatter Plot')
    #
    plt.xlabel('X')
    #
    plt.ylabel('Y')
    #
    # ax1.imshow(img)
    ax1.scatter(x,y,c = 'r',marker = 'o')
    
    #
    # plt.legend('x1')
    #
    plt.show()

def ReadFile(file_path):
    #read the .xyz file from file_path,and transfrom the file to a numpy
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

def ProdjectionModel(K,R_t,pts):
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

def Visual2dImg(uv,img):
    #plot scatter figure by uv coordinates
    x,y=uv[:,0],uv[:,1]
    # img=np.zeros((480,640,3),dtype=np.int)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #
    ax1.set_title('Scatter Plot')
    #
    plt.xlabel('X')
    #
    plt.ylabel('Y')
    #
    ax1.imshow(img)
    ax1.scatter(x,y,c = 'r',marker = 'o')
    
    #
    # plt.legend('x1')
    #
    plt.show()


def GenerateLabel(label_file,cls_color):
    #input the label of one image objects(mask)
    #use input to colorlize the per object different 
    #color for visualizing

    im_size=label_file.size
    I=np.zeros((im_size[1],im_size[0],3),dtype=np.uint8)
    # print(I.shape)
    label=np.array(label_file,dtype=np.int)
    class_num=22
    for i in range(0,class_num):
        x,y=np.where(label==i)
        # print(x.max())
        I[x[:],y[:],0]=cls_color[i,0]
        I[x[:],y[:],1]=cls_color[i,1]
        I[x[:],y[:],2]=cls_color[i,2]

    return I


def ReadObjFile(file_path):
    #read 3d model file .obj file 
    #return the vertices of 3d model:
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



def load_ply(path):
  """Loads a 3D mesh model from a PLY file.
  Copyright (c) 2019 Tomas Hodan
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
  
