import test
import os

def main(datapath):
    # function main
    ycb=test.YCB_V(datapath)

    model_path=ycb.path['model_path']

    data_path=ycb.path['data_path']

    # coco_json=test.COCO_M(ycb.path)

    model_data={}
    #sort the objects,[1,2,3...,21]
    _model_path=sorted(model_path,key=lambda x: int(x.split('/')[-1].split('_')[0]))

    #get model datas
    for i,path in enumerate(_model_path):
        #for ycb-v dataset]
        path=os.path.join(path,'textured.obj')
        # print(path)
        model=test.Model(path)

        model_data['{}'.format(i+1)]=model

    #get sample datas
    # imgs(name)
    # depth(name)
    # annotations(dict)
    # masks (name)
    ####

    # train_data=[]
    # for path in data_path:
    #     data_obj=test.DataPro(path)
    #     train_data.append(data_obj)
    # ##coo_json
    # print(len(train_data))
    coco_json=test.COCO_M(ycb.path)
    #initialize the categories
    coco_json.getcateg()
    ############################
    #for mark the imaga id  
    id=0
    for path in data_path:
        data_obj=test.DataPro(path)
        
        for i in range(len(data_obj.imgs)):
            #get the data
            name_img=data_obj.imgs[i]
            depth=data_obj.depth[i]
            ann=data_obj.ann[i]
            mask_name=data_obj.masks[i]
            coco_json.getimg(path,name_img,id)#for key:images
            coco_json.getann_per_img(path,model_data,name_img,id,ann) #for key:annotations

            id+=1
    coco_json.save_json(datapath)
    # print("Done!")



main('./YCB-Video')
