import xml.etree.ElementTree as xml


name_spliter = "_"
################################################################################

def mtgvidstore(xeg_doc):
    
    root = doc.getroot()
    for node in root.findall('node'):
        # for node of subMetamer scale, id is obviously not stands vid, 
        # but just new id assigned in mtg2rg during rootedgraph creating
        vid = node.attrib['id']
        org_name = node.attrib['name']
        new_name = org_name + name_spliter + str(vid)
        node.attrib['name'] = new_name

    return doc



def mtgvidrestore(xeg_fn):

    doc = xml.parse(xeg_fn)
    root = doc.getroot()
    for node in root.findall('node'):
        old_id = node.attrib['id']
        org_id = node.attrib['name'].split(name_spliter)[1] 
        if  old_id != org_id:
            node.attrib['id'] = org_id
            for edge in root.findall('edge'):
                if edge.attrib['src_id'] == old_id:
                    edge.attrib['src_id'] = org_id
                elif edge.attrib['dest_id'] == old_id:
                    edge.attrib['dest_id'] = org_id

    return doc
