from xml.dom.minidom import Document
import os
import shutil

def trainGen(rootPath, savePath):
    rootPath = rootPath + '/train'
    imageTrainPath = savePath + '/image_train'
    if not os.path.exists(imageTrainPath):
        os.makedirs(imageTrainPath)

    doc = Document()
    root = doc.createElement("TrainingImages")
    root.setAttribute('Version', '1.0')
    doc.appendChild(root)
    itemsInfo = doc.createElement('Items')
    itemsInfo.setAttribute('number', '')    # số lượng ảnh có trong dataset
    root.appendChild(itemsInfo)
    imageNum = 0
    scenceList = os.listdir(rootPath)
    
    for scenceName in scenceList:
        print('[INFO] Start to process {}'.format(scenceName))
        IDList = os.listdir(rootPath + '/' + scenceName)
        for IDName in IDList:
            imageList = os.listdir(rootPath + '/' + scenceName + '/' + IDName)
            for imageName in imageList:
                shutil.copyfile(rootPath + '/' + scenceName + '/' + IDName + '/' + imageName, imageTrainPath + '/' + imageName)
                cameraName = imageName.split('_')[0]
                nodeManager = doc.createElement('Item')
                nodeManager.setAttribute('imageName', imageName)
                nodeManager.setAttribute('personID', '{:04d}'.format(int(IDName)))
                nodeManager.setAttribute('cameraID', cameraName)
                nodeManager.setAttribute('sceneID', scenceName.split('_')[0])
                itemsInfo.appendChild(nodeManager)
                imageNum += 1
        print('[INFO] {} done!'.format(scenceName))
    
    itemsInfo.setAttribute('number', str(imageNum))
    fp = open(savePath + '/train_label.xml', 'w')
    doc.writexml(fp, indent="  ", addindent='\t', newl='\n', encoding='gb2312')
    fp.close()

def valGen(rootPath, savePath):
    rootPath = rootPath + '/validation'
    imageValPath = savePath + '/image_query'
    if not os.path.exists(imageValPath):
        os.makedirs(imageValPath)

    doc = Document()
    root = doc.createElement("ValidationImages")
    root.setAttribute('Version', '1.0')
    doc.appendChild(root)
    itemsInfo = doc.createElement('Items')
    itemsInfo.setAttribute('number', '')    # số lượng ảnh có trong dataset
    root.appendChild(itemsInfo)
    imageNum = 0
    scenceList = os.listdir(rootPath)

    for scenceName in scenceList:
        print('[INFO] Start to process {}'.format(scenceName))
        IDList = os.listdir(rootPath + '/' + scenceName)
        for IDName in IDList:
            imageList = os.listdir(rootPath + '/' + scenceName + '/' + IDName)
            for imageName in imageList:
                shutil.copyfile(rootPath + '/' + scenceName + '/' + IDName + '/' + imageName, imageValPath + '/' + imageName)
                cameraName = imageName.split('_')[0]
                nodeManager = doc.createElement('Item')
                nodeManager.setAttribute('imageName', imageName)
                nodeManager.setAttribute('personID', '{:04d}'.format(int(IDName)))
                nodeManager.setAttribute('cameraID', cameraName)
                nodeManager.setAttribute('sceneID', scenceName.split('_')[0])
                itemsInfo.appendChild(nodeManager)
                imageNum += 1
        print('[INFO] {} done!'.format(scenceName))
    
    itemsInfo.setAttribute('number', str(imageNum))
    fp = open(savePath + '/query_label.xml', 'w')
    doc.writexml(fp, indent="  ", addindent='\t', newl='\n', encoding='gb2312')
    fp.close()


if __name__ == '__main__':
    rootPath = '../data/AIC23_REID_DATA'
    savePath = '../data/AIC23_Track1_REID'

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    trainGen(rootPath, savePath)
    valGen(rootPath, savePath)