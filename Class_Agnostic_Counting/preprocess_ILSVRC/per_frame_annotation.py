import os
import os.path as osp
import sys
import xml.etree.ElementTree as ET

CLASS_IDS = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808', 'n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049'];

CLASS_NAMES = ['airplane','antelope','bear','bicycle','bird','bus','car','cattle','dog','domestic_cat',  'elephant','fox','giant_panda','hamster','horse','lion', 'lizard','monkey','motorcycle','rabbit', 'red_panda','sheep','snake','squirrel','tiger','train','turtle','watercraft','whale','zebra'];

class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})

def parse_xml(fp, new_fp3):
    tree = ET.parse(fp)
    root = tree.getroot()
    xmldict = XmlDictConfig(root)
    fp3_fd = new_fp3.split("/")
    with open(new_fp3, "w") as fp:
        if "object" in xmldict.keys():
            obj = xmldict["object"]
            obj_index = CLASS_IDS.index(obj['name'])
            fp.write(fp3_fd[-3])
            fp.write(" ")
            fp.write(fp3_fd[-2])
            fp.write(" ")
            fp.write(fp3_fd[-1])
            fp.write(" ")
            obj_track_id = obj["trackid"]
            fp.write(obj_track_id)
            fp.write(" ")
            obj_bbox = obj["bndbox"]
            x_max, x_min, y_max, y_min = int(obj_bbox['xmax']), int(obj_bbox['xmin']), int(obj_bbox['ymax']), int(obj_bbox['ymax'])
            obj_w, obj_h = x_max-x_min+1, y_max-y_min+1
            fp.write(str(x_min) +  " " + str(y_min) + " " + str(obj_w) + " " + str(obj_h))
            fp.write(" ")
        else:
            fp.write(fp3_fd[-3])
            fp.write(" ")
            fp.write(fp3_fd[-2])
            fp.write(" ")
            fp.write(fp3_fd[-1])
            fp.write(" ")
        size = xmldict["size"]
        fp.write(size["width"])
        fp.write(" ")
        fp.write(size["height"])
        fp.write(" ")
        fp.write(fp3)
        fp.write("\n")
 
def mkdir(p):
    if not osp.isdir(p):
        os.mkdir(p)
 
if __name__=="__main__":
    i = 1
    folder = "../../ILSVRC2015/Annotations/VID/"
    new_folder = "../../ILSVRC2015/Annotations/curated_VID/"
    if not osp.isdir(new_folder):
        os.mkdir(new_folder)
    
    dir_1 = next(os.walk(folder))[1]
    for each_dir in dir_1:
        fp1 = osp.join(folder, each_dir)
        new_fp1 = osp.join(new_folder, each_dir)
        mkdir(new_fp1)
        dir2 = next(os.walk(fp1))[1]
        for each_dir_2 in dir2:
            fp2 = osp.join(fp1, each_dir_2)
            new_fp2 = osp.join(new_fp1, each_dir_2)
            mkdir(new_fp2)
            dir3 = next(os.walk(fp2))[2]
            for each_dir_3 in dir3:
                fp3 = osp.join(fp2, each_dir_3)
                new_each_dir_3 = str(each_dir_3)[:-4] + ".txt"
                new_fp3 = osp.join(new_fp2, new_each_dir_3)
                parse_xml(fp3, new_fp3)
