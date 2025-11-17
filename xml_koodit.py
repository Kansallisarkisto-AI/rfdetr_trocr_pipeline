from bs4 import BeautifulSoup as bs
from yattag import indent
from pathlib import Path
import os
from lxml import etree
import datetime

class PageXML:

    def save_page(self, xml_file, path):
        """Saves Page xml file."""
        xml_file.write(path, xml_declaration=True, encoding='utf-8', method="xml")
        print('Page XML file saved to ', path)

    def format_polygon(self, polygon):
        """Formats polygon from a list of lists into a string."""
        polygon_str = ''
        for pair in polygon:
            polygon_str += '%s,%s '%(int(pair[0]), int(pair[1]))
        return polygon_str

    def create_xml(self, data, image_path): 
        attr_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance", "schemaLocation")
        root = etree.Element('PcGts',
                            {attr_qname: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd'},
                            nsmap={None: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
                                    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
                                    })

        # Metadata
        m = etree.SubElement(root, "Metadata")
        cr = etree.SubElement(m, "Creator")
        cr.text = 'Kansallisarkisto - National Archives of Finland'

        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f+02:00')
        cd = etree.SubElement(m, "Created")
        cd.text = now
        lc = etree.SubElement(m,"LastChange")
        lc.text = now

        #Root -> Page
        p = etree.SubElement(root, "Page")
        p.set('imageWidth', str(data[0]['width']))
        p.set('imageHeight', str(data[0]['height']))
        p.set('imageFilename', data[0]['img_name'])

        #Page -> ReadingOrder
        ro = etree.SubElement(p, "ReadingOrder")
        #ReadingOrder -> OrderedGroup
        og = etree.SubElement(ro, "OrderedGroup") 
        og_id = 'ro'
        og.set('id', og_id)
        og.set('caption', "Regions reading order")
        
        # Loop over regions
        for i, region_dict in enumerate(data):
            region_id = 'r' + str(i)
            region_name = region_dict['region_name']
            region_polygon_coords = self.format_polygon(region_dict['region_coords'])
            custom_region_tag = "readingOrder {index:%s;} structure {type:%s;}" %(str(i), region_name)

            #OrderedGroup -> RegionRefIndexed
            rri = etree.SubElement(og, "RegionRefIndexed")
            rri.set('index', str(i))
            rri.set('regionRef', region_id)

            # Page -> TextRegion
            tr = etree.SubElement(p, "TextRegion")
            tr.set('id', region_id)
            tr.set('type', region_name)
            tr.set('custom', custom_region_tag)
            
            #TextRegion -> CSoords
            region_coords = etree.SubElement(tr,"Coords")
            region_coords.set('points', region_polygon_coords)
            
            # Loop over text lines belonging to the region
            for j, line_dict in enumerate(region_dict['text_lines']):
                line_id = region_id + 'l' + str(j)
                line_text = line_dict['text']
                line_polygon_coords = self.format_polygon(line_dict['polygon'])
                custom_line_tag = "readingOrder {index:%s;}" % str(j)

                #TextRegion -> TextLine
                tl = etree.SubElement(tr, "TextLine")
                tl.set('id', line_id)
                tl.set('custom', custom_line_tag)
                
                #TextLine -> Coords
                line_coords = etree.SubElement(tl, "Coords")
                line_coords.set('points', line_polygon_coords)

                #TextLine -> TextEquiv
                te = etree.SubElement(tl, "TextEquiv")
                #TextEquiv -> Unicode
                uc = etree.SubElement(te, "Unicode")
                uc.text = line_text.strip()
                
            xml_doc = etree.ElementTree(root)

        return xml_doc
    
    def get_page(self, page_dict, image_path, save_path):
        page = self.create_xml(page_dict, image_path)
        self.save_page(page, save_path)


class AltoXML:
    def __init__(self, seg_model, line_model, htr_model):
        self.seg_model = seg_model
        self.line_model = line_model
        self.htr_model = htr_model
    
    def save_alto(self, newsoup, path):
        """Saves Alto xml file."""
        with open(path,"w") as f: 
            f.write(indent(str(newsoup))) 
        print('XML file saved to ', path)

    def format_polygon(self, polygon):
        """Formats polygon from a list of lists into a string."""
        polygon_str = ''
        for pair in polygon:
            polygon_str += '%s,%s '%(int(pair[0]), int(pair[1]))
        return polygon_str
    
    def get_region_ids(self, data):
        """Creates region ids based on the region names."""
        region_names = [region_dict['region_name'] for region_dict in data]
        unique_names = list(set(region_names))
        ind_dict = {name: 0 for name in unique_names}
        new_names = []
        for i, name in enumerate(region_names):
            ind = ind_dict[name]
            new_name = name + '_' + str(ind)
            new_names.append(new_name)
            ind_dict[name] = ind_dict.get(name, 0) + 1
        return new_names


    def create_xml(self, data, image_path):
        """Transports the text line polygons and predicted text content
        into Alto xml format."""
        # xml template where to start building the Alto xml            
        newsoup=bs(f"""
        <alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xmlns="http://www.loc.gov/standards/alto/ns-v4#"
        xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-0.xsd"
        xmlns:xlink="http://www.w3.org/1999/xlink">
        <Description>
            <MeasurementUnit>mm10</MeasurementUnit>
            <sourceImageInformation>
                <fileName>{image_path}</fileName>
            </sourceImageInformation>
            <Processing ID="TEXT_REGION_SEGMENTATION">
                <processingAgency>Kansallisarkisto - National Archives of Finland</processingAgency>
                <processingStepDescription>Text region detection</processingStepDescription>
                <processingSoftware>
                    <softwareCreator>Roboflow</softwareCreator>
                    <softwareName>RFDETRSegPreview</softwareName>
                    <softwareVersion>{self.seg_model}</softwareVersion>
                </processingSoftware>
            </Processing>
            <Processing ID="TEXT_LINE_SEGMENTATION">
                <processingAgency>Kansallisarkisto - National Archives of Finland</processingAgency>
                <processingStepDescription>Text line detection</processingStepDescription>
                <processingSoftware>
                    <softwareCreator>Roboflow</softwareCreator>
                    <softwareName>RFDETRSegPreview</softwareName>
                    <softwareVersion>{self.line_model}</softwareVersion>
                </processingSoftware>
            </Processing>
            <Processing ID="TEXT_RECOGNITION">
                <processingAgency>Kansallisarkisto - National Archives of Finland</processingAgency>
                <processingStepDescription>Handwritten text recognition</processingStepDescription>
                <processingSoftware>
                    <softwareCreator>Microsoft</softwareCreator>
                    <softwareName>TrOCR</softwareName>
                    <softwareVersion>{self.htr_model}</softwareVersion>
                </processingSoftware>
            </Processing>
        </Description>
        <Styles></Styles>
        <Layout>        
        </Layout>
        </alto>
        ""","xml")
        
        # create Page element
        new_tag_page=newsoup.new_tag("Page", 
                ID=data[0]['img_name'],
                WIDTH=data[0]['width'],
                HEIGHT=data[0]['height'],
                PC="{0:.2f}".format(data[0]['page_conf_mean']),
                PC50="{0:.2f}".format(data[0]['page_conf_median']),
                PC25="{0:.2f}".format(data[0]['page_conf_25']),
                PC75="{0:.2f}".format(data[0]['page_conf_75']),
                N_LONG_ROWTEXT=data[0]['n_long_rowtext'],
                LANGUAGE=data[0]['language'],
                PHYSICAL_IMG_NR=0)
        
        new_tag_printspace=newsoup.new_tag("PrintSpace", 
                HPOS=0,VPOS=0,
                WIDTH=data[0]['width'],
                HEIGHT=data[0]['height'])
        
        new_tag_page.append(new_tag_printspace)    
        newsoup.Layout.append(new_tag_page)
        
        # add elements to the page
        text_blocks=bs('',"html.parser")
        region_ids = self.get_region_ids(data)
        # Loop over detected regions
        for i, region_dict in enumerate(data):
            # Create TextBlock element for each region
            text_block=newsoup.new_tag("TextBlock", ID=region_ids[i])
            # Add text region polygon values to Shape tag of the text block
            region_shape=newsoup.new_tag("Shape")
            region_polygon_str = self.format_polygon(region_dict['region_coords'])
            region_polygon = newsoup.new_tag("Polygon", POINTS=region_polygon_str)
            region_shape.append(region_polygon)
            text_block.append(region_shape)
            # Loop over text lines belonging to the region
            for j, line_dict in enumerate(region_dict['text_lines']):
                line_id = region_ids[i] + '_line_' + str(j)
                # Create TextLine element for each detected text line
                text_line=newsoup.new_tag("TextLine", ID=line_id)
                text_string=newsoup.new_tag("String", ID=line_id, CONTENT=line_dict['text'], RL=line_dict['row_length'], WC=str(round(line_dict['text_conf'],2)))
                # Add text line polygon values to Shape tag of the text line
                line_shape=newsoup.new_tag("Shape")
                line_polygon_str = self.format_polygon(line_dict['polygon'])
                line_polygon = newsoup.new_tag("Polygon", POINTS=line_polygon_str)
                line_shape.append(line_polygon)
                text_line.append(line_shape)
                text_line.append(text_string)
                text_block.append(text_line)
            
            text_blocks.append(text_block)
        
        newsoup.PrintSpace.append(text_blocks)

        return newsoup

    def get_alto(self, page_dict, image_path, save_path):
        alto = self.create_xml(page_dict, image_path)
        self.save_alto(alto, save_path)


def get_xml(text_predictions, input_data):
    """Function for saving the results in xml file.
    
    Save OCR results to XML file in PAGE and/or ALTO format.

    Args:
        text_predictions: The OCR text predictions to be saved.
        input_data: Object containing configuration parameters including:
            - image_path: Path to the input image file.
            - page_xml: Boolean flag to enable PAGE XML output.
            - alto_xml: Boolean flag to enable ALTO XML output.
            - xml_path: Base directory path for saving XML files.
            - region_segment_model: Model used for region segmentation (for ALTO).
            - line_segment_model: Model used for line segmentation (for ALTO).
            - text_recognition_model: Model used for text recognition (for ALTO).

    Returns:
        None. XML files are saved to disk in subdirectories ('page' and/or 'alto')
        within the specified xml_path.
    """
    xml_name = Path(input_data.image_path).stem + '.xml'
    # PAGE XML option
    if input_data.page_xml and input_data.xml_path:
        page_maker = PageXML()
        save_folder = Path(input_data.xml_path, 'page')
        os.makedirs(save_folder, exist_ok=True)
        save_path = str(Path(save_folder, xml_name))
        page_maker.get_page(text_predictions, input_data.image_path, save_path)
    # Alto XML option
    if input_data.alto_xml and input_data.xml_path:
        save_folder = Path(input_data.xml_path, 'alto')
        os.makedirs(save_folder, exist_ok=True)
        save_path = str(Path(save_folder, xml_name))
        alto_maker = AltoXML(input_data.region_segment_model_name, input_data.line_segment_model_name, input_data.text_recognition_model_name)
        alto_maker.get_alto(text_predictions, input_data.image_path, save_path)
