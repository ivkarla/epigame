import re
import numpy as np
import csv
import ast

def anatomicREL(tag):
    region = ['Unknown',
               'Left-Cerebral-White-Matter',
               'Left-Cerebral-Cortex',
               'Left-Lateral-Ventricle',
               'Left-Inf-Lat-Vent',
               'Left-Cerebellum-White-Matter',
               'Left-Cerebellum-Cortex',
               'Left-Thalamus-Proper',
               'Left-Caudate',
               'Left-Putamen',
               'Left-Pallidum',
               '3rd-Ventricle',
               '4th-Ventricle',
               'Brain-Stem',
               'Left-Hippocampus',
               'Left-Amygdala',
               'CSF',
               'Left-Accumbens-area',
               'Left-VentralDC',
               'Left-vessel',
               'Left-choroid-plexus',
               'Right-Cerebral-White-Matter',
               'Right-Cerebral-Cortex',
               'Right-Lateral-Ventricle',
               'Right-Inf-Lat-Vent',
               'Right-Cerebellum-White-Matter',
               'Right-Cerebellum-Cortex',
               'Right-Thalamus-Proper',
               'Right-Caudate',
               'Right-Putamen',
               'Right-Pallidum',
               'Right-Hippocampus',
               'Right-Amygdala',
               'Right-Accumbens-area',
               'Right-VentralDC',
               'Right-vessel',
               'Right-choroid-plexus',
               '5th-Ventricle',
               'WM-hypointensities',
               'non-WM-hypointensities',
               'Optic-Chiasm',
               'CC Posterior',
               'CC Mid Posterior',
               'CC Central',
               'CC Mid Anterior',
               'CC Anterior']
    indx = [0,
            2,
            3,
            4,
            5,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            24,
            26,
            28,
            30,
            31,
            41,
            42,
            43,
            44,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            54,
            58,
            60,
            62,
            63,
            72,
            77,
            80,
            85,
            251,
            252,
            253,
            254,
            255] 
    anatomic = region[indx.index(tag)]
    return anatomic

def RAStoIJK(ras,volumeNode):
    transformRasToVolumeRas = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
    point_VolumeRas = transformRasToVolumeRas.TransformPoint(ras[0:3])
    volumeRasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
    point_Ijk = [0, 0, 0, 1]
    volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
    point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]
    return point_Ijk

def insideRES(nodeRAS):
    segmentationNode = getNode('Segmentation')
    try : 
        segmentLabelmapNode = getNode('res_map')
    except: 
        segmentLabelmapNode = slicer.vtkMRMLLabelMapVolumeNode()
        segmentLabelmapNode.SetName('res_map')
    slicer.mrmlScene.AddNode(segmentLabelmapNode)
    segmentIDs = vtk.vtkStringArray()
    segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    referenceNode = segmentationNode.GetNodeReference(slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, segmentIDs, segmentLabelmapNode, referenceNode)   
    voxelArray = slicer.util.arrayFromVolume(segmentLabelmapNode)   
    point_Ijk = RAStoIJK(nodeRAS,segmentLabelmapNode)    
    try: 
        inside_val = voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]] # return 1 when the node is in the resection, 0 when not 
    except: 
        inside_val = 0    
    return inside_val

def distance2res(nodeRAS):
    step = 0.05
    if insideRES(nodeRAS) == 1:
        distance = 0
    else:
        searching = True
        segmentationNode = getNode('Segmentation')
        segment = segmentationNode.GetSegmentation().GetNthSegment(0)
        pd = segment.GetRepresentation('Closed surface')
        com = vtk.vtkCenterOfMass()
        com.SetInputData(pd)
        com.Update()
        center = com.GetCenter()
        i = 1
        while searching:
            next_point_R = nodeRAS[0] + i*step*(center[0]-nodeRAS[0])  
            next_point_A = nodeRAS[1] + i*step*(center[1]-nodeRAS[1])
            next_point_S = nodeRAS[2] + i*step*(center[2]-nodeRAS[2])
            next_point = [next_point_R,next_point_A,next_point_S]
            i = i+1
            if insideRES(next_point) == 1: 
                searching = False
        distance = np.sqrt( (next_point[0]-nodeRAS[0])**2 + (next_point[1]-nodeRAS[1])**2 + (next_point[2]-nodeRAS[2])**2 )
    return distance

def showNodes(selected,markup_list='onset'):   
    '''computes the ratio of network nodes (selected) within resection (Score1) and mean distance of nodes from the resection (Score2)'''    
    # Selected labels/nodes/channels
    labels = []
    done_labels = []
    pairs = []
    for pair in selected:
        labels.append(pair.split('-')[0])
        labels.append(pair.split('-')[1])
        pairs.append(pair.split('-'))
    
    # Select rulers
    annotationHierarchyNode = getNode('Ruler List') # rulers
    children = annotationHierarchyNode.GetNumberOfChildrenNodes() # number of rulers
    
    # Select the fiducials
    try:
        fidNode = getNode(markup_list)
    except:
        fidNode = slicer.vtkMRMLMarkupsFiducialNode()
        fidNode.SetName(markup_list)
        slicer.mrmlScene.AddNode(fidNode)
    
    # Sleect the volume
    volumeNode = getNode('aseg')
    voxelArray = slicer.util.arrayFromVolume(getNode('aseg'))
    
    # Initialize Onset list
    onset = [['Node','R','A','S','i','j','k','Anatomical Label','Anatomic Region','Inside Resection','Distance']]
    
    for ruler_index in range(children):
        annotation = annotationHierarchyNode.GetNthChildNode(ruler_index).GetAssociatedNode() # one ruler
        if annotation != None:
            name = annotation.GetName() # name of the ruler e.g. 'A'
            print('for electrode named ', name, 'in slicer, checking:\n')
            for label in labels: 
                if label in done_labels:
                    pass
                else:
                    num = int(re.search(r'\d+', label).group()) # extract the number
                    if label[1].isalpha(): 
                        letter = label[0:2]
                    else:
                        letter = label[0:2] if "'" in label else re.search(r'\w', label).group()
                    print('node', letter)
                    if letter == name:
                        
                        if num == 1: # if the number is one we just have to select the start of the ruler
                            pstart = [0,0,0]
                            annotation.GetPosition1(pstart) # pstart now has the coordinates of the point 
                            fidNode.AddFiducialFromArray(pstart,letter+str(num))
                            ras = pstart
                            point_Ijk = RAStoIJK(ras,volumeNode)
                            # onset.append([letter+str(num),ras[0],ras[1],ras[2],point_Ijk[0],point_Ijk[1],point_Ijk[2],voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]],anatomicREL(voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]]),insideRES(ras)])
                            onset.append([letter+str(num),ras[0],ras[1],ras[2],point_Ijk[0],point_Ijk[1],point_Ijk[2],voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]],anatomicREL(voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]]),insideRES(ras),distance2res(ras)])                            done_labels.append(letter+str(num))
                        else:
                            pstart = [0,0,0]
                            annotation.GetPosition1(pstart) 
                            pend = [0,0,0]
                            annotation.GetPosition2(pend) 
                            measure = annotation.GetDistanceMeasurement()
                            location_x = pstart[0] + (num-1)*(3.5/measure)*(pend[0]-pstart[0])
                            location_y = pstart[1] + (num-1)*(3.5/measure)*(pend[1]-pstart[1])
                            location_z = pstart[2] + (num-1)*(3.5/measure)*(pend[2]-pstart[2])
                            location = [location_x,location_y,location_z]
                            fidNode.AddFiducialFromArray(location,letter+str(num))
                            ras = location
                            point_Ijk = RAStoIJK(ras,volumeNode)
                            # onset.append([letter+str(num),ras[0],ras[1],ras[2],point_Ijk[0],point_Ijk[1],point_Ijk[2],voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]],anatomicREL(voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]]),insideRES(ras)])
                            onset.append([letter+str(num),ras[0],ras[1],ras[2],point_Ijk[0],point_Ijk[1],point_Ijk[2],voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]],anatomicREL(voxelArray[point_Ijk[2],point_Ijk[1],point_Ijk[0]]),insideRES(ras),distance2res(ras)])
                            done_labels.append(letter+str(num))
    inres = []   
    for subject in range(1,len(onset)):
        inres.append(onset[subject][9])
        distances.append(onset[subject][10])
    mean_dist = np.mean(distances)
    norm_dist = mean_dist/max(distances)
    inresrate = np.mean(inres)
    return inresrate, mean_dist, onset



def AnRes(name,source,out):
    '''takes a two-column CSV file with the analysis results (method:list of selected nodes) located at source;
    computes the ratio of network nodes within resection (Score1) and mean distance of nodes from the resection (Score2) for all methods;
    saves results as a CSV file at out path'''
    with open(source,'r') as f:
        reader = csv.reader(f)
        nodes = list(reader)
    results = [['Method','InResRate','MeanDist']] 
    for method in range(len(nodes)):
        method_name = nodes[method][0]
        print(method_name)
        selected = ast.literal_eval(nodes[method][-1])
        print(selected)
        inres, mean_dist, onset = showNodes(selected)
        results.append([method_name, inres, mean_dist])
        onset = np.array(onset).T.tolist()
        with open(out+name+'_'+method_name+'_nodes_info'+'.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(onset)
    results = np.array(results).T.tolist()
    with open(out+name+'.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
